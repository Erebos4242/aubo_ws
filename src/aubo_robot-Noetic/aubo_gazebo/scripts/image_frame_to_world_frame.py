x_per_pixel = 0.5 / 541
y_per_pixel = 0.5 / 541
image_w = 1920
image_h = 1080
height = 1.5342424

tan_w = x_per_pixel * image_w / 2 / height
tan_h = y_per_pixel * image_h / 2 / height


def w_h_per_pixel(h):
    w_per_pixel = (tan_w * h * 2) / 1920
    h_per_pixel = (tan_h * h * 2) / 1080
    return w_per_pixel, h_per_pixel


def image_frame_to_world_frame(h, u, v):
    w_per_pixel, h_per_pixel = w_h_per_pixel(h)
    x = (image_w / 2 - v) * w_per_pixel
    y = (image_h / 2 - u) * h_per_pixel
    return y, x, 2.5 - h
