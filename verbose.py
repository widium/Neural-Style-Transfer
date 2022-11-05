import tensorflow as tf
import numpy as np

def print_features_maps_style(style_target):
    print("Style : ")
    for name, output in style_target.items():
        print(" ==", name, " ==")
        print("  shape: ", output.numpy().shape)
        print("  min: ", output.numpy().min())
        print("  max: ", output.numpy().max())
        print("  mean: ", output.numpy().mean())

def print_features_maps_content(content_target):
    print("Content : ")
    for name, output in content_target.items():
        print(" ==", name, " ==")
        print("  shape: ", output.numpy().shape)
        print("  min: ", output.numpy().min())
        print("  max: ", output.numpy().max())
        print("  mean: ", output.numpy().mean())