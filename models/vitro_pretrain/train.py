from models.backbone.ResMobileNet import ResMobileNet, res_mobilenet_conf

inverted_residual_setting, last_channel = res_mobilenet_conf(
    width_mult=1.0
)

model = ResMobileNet(
    inverted_residual_setting=inverted_residual_setting,
    last_channel=last_channel
)

print(model)