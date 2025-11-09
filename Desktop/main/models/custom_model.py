import torch
import torch.nn as nn
from torchvision.models import resnet50
import torch_pruning as pruning


class CustomDetectionModel(nn.Module):
    """
    Custom object detection model based on ResNet backbone with detection heads.
    """

    def __init__(self, num_classes, backbone='resnet50'):
        """
        Initialize the model.

        Args:
            num_classes (int): Number of classes to detect.
            backbone (str): Backbone architecture to use.
        """
        super(CustomDetectionModel, self).__init__()
        self.num_classes = num_classes

        # Load pretrained backbone
        if backbone == 'resnet50':
            self.backbone = resnet50(pretrained=True)
            self.backbone_features = nn.Sequential(*list(self.backbone.children())[:-2])
            self.feature_channels = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Detection head
        self.det_head = nn.Sequential(
            nn.Conv2d(self.feature_channels, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # Classification branch
        self.cls_head = nn.Conv2d(256, num_classes, kernel_size=1)

        # Regression branch (bbox coordinates)
        self.reg_head = nn.Conv2d(256, 4, kernel_size=1)

        # Objectness branch
        self.obj_head = nn.Conv2d(256, 1, kernel_size=1)

    def forward(self, x):
        """
        Forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            dict: Dictionary containing predictions.
        """
        # Extract features
        features = self.backbone_features(x)  # Shape: (batch, 2048, H/32, W/32)

        # Detection head
        det_features = self.det_head(features)  # Shape: (batch, 256, H/32, W/32)

        # Predictions
        cls_pred = self.cls_head(det_features)  # Shape: (batch, num_classes, H/32, W/32)
        reg_pred = self.reg_head(det_features)  # Shape: (batch, 4, H/32, W/32)
        reg_pred = reg_pred.clone()
        reg_pred[0] = torch.sigmoid(reg_pred[0])  # tx
        reg_pred[1] = torch.sigmoid(reg_pred[1])  # ty
        obj_pred = self.obj_head(det_features)  # Shape: (batch, 1, H/32, W/32)

        return {
            'cls_pred': cls_pred,
            'reg_pred': reg_pred,
            'obj_pred': obj_pred,
            'features': features
        }

    def prune_model(self, pruning_ratio=0.3, example_input=None):
        """
        Apply structured pruning to the model.

        Args:
            pruning_ratio (float): Ratio of parameters to prune (0.0-1.0)
            example_input (torch.Tensor): Example input for pruning analysis

        Returns:
            dict: Pruning statistics
        """
        if example_input is None:
            # Create dummy input for analysis
            example_input = torch.randn(1, 3, 640, 640)

        # Set up pruning
        model = self
        example_input = example_input.to(next(model.parameters()).device)

        # Define pruning strategy
        strategy = pruning.strategy.L1Strategy()  # Prune by L1 norm

        # Prune detection head layers
        for module_name, module in model.named_modules():
            if isinstance(module, nn.Conv2d) and 'det_head' in module_name:
                pruning_info = pruning.get_pruning_info(module, example_input, strategy)
                if pruning_info['prunable']:
                    pruning.prune_conv2d(module, example_input, strategy, pruning_ratio)

        # Update batch norm layers after pruning
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                pruning.prune_batchnorm2d(module)

        # Calculate pruning statistics
        total_params = sum(p.numel() for p in model.parameters())
        nonzero_params = sum(torch.count_nonzero(p).item() for p in model.parameters())
        sparsity = 1.0 - (nonzero_params / total_params)

        return {
            'total_params': total_params,
            'nonzero_params': nonzero_params,
            'sparsity': sparsity,
            'pruning_ratio': pruning_ratio
        }

    def quantize_model(self, calibration_data=None):
        """
        Apply dynamic quantization to the model.

        Args:
            calibration_data (torch.Tensor): Data for calibration (optional for dynamic quantization)

        Returns:
            torch.nn.Module: Quantized model
        """
        # Prepare model for quantization
        model = self

        # Fuse Conv2d + BatchNorm2d + ReLU layers
        model.eval()
        fused_model = torch.quantization.fuse_modules(
            model,
            [['det_head.0', 'det_head.1', 'det_head.2'],  # Conv2d + BatchNorm2d + ReLU
             ['det_head.3', 'det_head.4', 'det_head.5']],  # Conv2d + BatchNorm2d + ReLU
            inplace=True
        )

        # Specify quantization configuration
        fused_model.qconfig = torch.quantization.get_default_qconfig('fbgemm')

        # Prepare for quantization
        torch.quantization.prepare(fused_model, inplace=True)

        # Calibrate with data if provided
        if calibration_data is not None:
            with torch.no_grad():
                for batch in calibration_data:
                    fused_model(batch)

        # Convert to quantized model
        quantized_model = torch.quantization.convert(fused_model, inplace=True)

        return quantized_model


class YOLOLoss(nn.Module):
    """
    YOLO-style loss function for object detection.
    """

    def __init__(self, num_classes, lambda_coord=5.0, lambda_noobj=0.5):
        """
        Initialize the loss function.

        Args:
            num_classes (int): Number of classes.
            lambda_coord (float): Weight for coordinate loss.
            lambda_noobj (float): Weight for no-object loss.
        """
        super(YOLOLoss, self).__init__()
        self.num_classes = num_classes
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj

        self.mse_loss = nn.MSELoss(reduction='sum')
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='sum')

    def forward(self, predictions, targets):
        """
        Compute the loss.

        Args:
            predictions (dict): Model predictions.
            targets (list): List of target tensors.

        Returns:
            dict: Dictionary containing loss components.
        """
        cls_pred = predictions['cls_pred']
        reg_pred = predictions['reg_pred']
        obj_pred = predictions['obj_pred']

        batch_size = cls_pred.size(0)
        grid_size = cls_pred.size(-1)

        total_loss = 0
        obj_loss = 0
        noobj_loss = 0
        coord_loss = 0
        cls_loss = 0

        for b in range(batch_size):
            target = targets[b]  # Shape: (num_objects, 5) [class, x, y, w, h]

            # Create target grids
            obj_mask = torch.zeros((grid_size, grid_size), device=cls_pred.device)
            noobj_mask = torch.ones((grid_size, grid_size), device=cls_pred.device)
            cls_target = torch.zeros((self.num_classes, grid_size, grid_size), device=cls_pred.device)
            reg_target = torch.zeros((4, grid_size, grid_size), device=cls_pred.device)

            if len(target) > 0:
                for obj in target:
                    class_id, x, y, w, h = obj

                    # Convert to grid coordinates
                    grid_x = int(x * grid_size)
                    grid_y = int(y * grid_size)

                    if 0 <= grid_x < grid_size and 0 <= grid_y < grid_size:
                        obj_mask[grid_y, grid_x] = 1
                        noobj_mask[grid_y, grid_x] = 0

                        cls_target[int(class_id), grid_y, grid_x] = 1
                        reg_target[0, grid_y, grid_x] = x * grid_size - grid_x  # dx
                        reg_target[1, grid_y, grid_x] = y * grid_size - grid_y  # dy
                        reg_target[2, grid_y, grid_x] = torch.sqrt(w)  # sqrt(w)
                        reg_target[3, grid_y, grid_x] = torch.sqrt(h)  # sqrt(h)

            # Objectness loss
            obj_pred_b = obj_pred[b, 0]  # Shape: (grid_size, grid_size)
            obj_loss += self.bce_loss(obj_pred_b, obj_mask)

            # No-object loss
            noobj_loss += self.lambda_noobj * self.bce_loss(obj_pred_b * (1 - obj_mask), noobj_mask * 0)

            # Coordinate loss
            coord_loss += self.lambda_coord * self.mse_loss(reg_pred[b] * obj_mask.unsqueeze(0), reg_target)

            # Classification loss
            cls_loss += self.bce_loss(cls_pred[b] * obj_mask.unsqueeze(0), cls_target)

        total_loss = obj_loss + noobj_loss + coord_loss + cls_loss

        return {
            'total_loss': total_loss / batch_size,
            'obj_loss': obj_loss / batch_size,
            'noobj_loss': noobj_loss / batch_size,
            'coord_loss': coord_loss / batch_size,
            'cls_loss': cls_loss / batch_size
        }
