from autoaugment import SubPolicy

policies = [
        SubPolicy(0.1, "invert", 7),
        SubPolicy(0.2, "contrast", 6),
        SubPolicy(0.7, "rotate", 2),
        SubPolicy(0.3, "translateX", 9),
        SubPolicy(0.8, "sharpness", 1),

        SubPolicy(0.9, "sharpness", 3),
        SubPolicy(0.5, "shearY", 2),
        SubPolicy(0.7, "translateY", 2) ,
        SubPolicy(0.5, "autocontrast", 5),
        SubPolicy(0.9, "equalize", 2), #

        SubPolicy(0.2, "shearY", 5),
        SubPolicy(0.3, "posterize", 5), #
        SubPolicy(0.4, "color", 3),
        SubPolicy(0.6, "brightness", 5), #
        SubPolicy(0.3, "sharpness", 9),

        SubPolicy(0.7, "brightness", 9),
        SubPolicy(0.6, "equalize", 5),
        SubPolicy(0.5, "equalize", 1),
        SubPolicy(0.6, "contrast", 7),
        SubPolicy(0.6, "sharpness", 5),
        
        SubPolicy(0.7, "color", 5),
        SubPolicy(0.5, "translateX", 5), #
        SubPolicy(0.3, "equalize", 7),
        SubPolicy(0.4, "autocontrast", 8),
        SubPolicy(0.4, "translateY", 3),
        SubPolicy(0.2, "sharpness", 6),
        SubPolicy(0.9, "brightness", 6),
        SubPolicy(0.2, "color", 8),
        SubPolicy(0.5, "solarize", 0),
        SubPolicy(0.0, "invert", 0), #
        SubPolicy(0.2, "equalize", 0),
        SubPolicy(0.6, "autocontrast", 0), #
        SubPolicy(0.2, "equalize", 8),
        SubPolicy(0.6, "equalize", 4),
        SubPolicy(0.9, "color", 5),
        SubPolicy(0.6, "equalize", 5), #
        SubPolicy(0.8, "autocontrast", 4),
        SubPolicy(0.2, "solarize", 4), #
        SubPolicy(0.1, "brightness", 3),
        SubPolicy(0.7, "color", 0),
        SubPolicy(0.4, "solarize", 1),
        SubPolicy(0.9, "autocontrast", 0), #
        SubPolicy(0.9, "translateY", 3),
        SubPolicy(0.7, "translateY", 3), #
        SubPolicy(0.9, "autocontrast", 1),
        SubPolicy(0.8, "solarize", 1), #
        SubPolicy(0.8, "equalize", 5),
        SubPolicy(0.1, "invert", 0),  #
        SubPolicy(0.7, "translateY", 3),
        SubPolicy(0.9, "autocontrast", 1),
        ]
        

policies_replace = [
        SubPolicy(0.1, "invert", 7),
        SubPolicy(0.2, "contrast", 6),
        SubPolicy(0.7, "rotate", 2),
        SubPolicy(0.3, "translateX", 9),
        SubPolicy(0.8, "sharpness", 1),

        SubPolicy(0.9, "sharpness", 3),
        SubPolicy(0.5, "shearY", 2),
        SubPolicy(0.7, "translateY", 2) ,
        SubPolicy(0.5, "autocontrast", 5),
        SubPolicy(0.9, "equalize", 2), #

        SubPolicy(0.2, "shearY", 5),
        SubPolicy(0.3, "posterize", 5), #
        SubPolicy(0.4, "color", 3),
        SubPolicy(0.6, "brightness", 5), #
        SubPolicy(0.3, "sharpness", 9),

        SubPolicy(0.7, "brightness", 9),
        SubPolicy(0.6, "translateX", 4),
        SubPolicy(0.5, "equalize", 1),
        SubPolicy(0.6, "contrast", 7),
        SubPolicy(0.6, "sharpness", 5),
        
        SubPolicy(0.7, "color", 5),
        SubPolicy(0.5, "translateX", 5), #
        SubPolicy(0.3, "equalize", 7),
        SubPolicy(0.4, "autocontrast", 8),
        SubPolicy(0.4, "translateY", 5), #translateY 3

        SubPolicy(0.2, "sharpness", 6),
        SubPolicy(0.9, "brightness", 6),
        SubPolicy(0.2, "color", 8),
        SubPolicy(0.5, "solarize", 0),
        SubPolicy(0.0, "invert", 0), #

        SubPolicy(0.2, "equalize", 0),
        SubPolicy(0.6, "autocontrast", 0), #
        SubPolicy(0.2, "equalize", 8),
        SubPolicy(0.6, "equalize", 4),
        SubPolicy(0.9, "color", 5),

        SubPolicy(0.6, "equalize", 5), #
        SubPolicy(0.8, "autocontrast", 4),
        SubPolicy(0.2, "solarize", 4), #
        SubPolicy(0.1, "brightness", 3),
        SubPolicy(0.7, "color", 0),

        SubPolicy(0.4, "solarize", 1),
        SubPolicy(0.9, "autocontrast", 0), #
        SubPolicy(0.9, "translateY", 3),
        SubPolicy(0.7, "translateY", 4), #
        SubPolicy(0.9, "autocontrast", 1),
        
        SubPolicy(0.8, "solarize", 1), #
        SubPolicy(0.8, "equalize", 5),
        SubPolicy(0.1, "invert", 0),  #
        SubPolicy(0.7, "translateY", 3),
        SubPolicy(0.9, "autocontrast", 1),
        ]

policies_names = [
        "invert_7",
        "contrast_6",
        "rotate_2",
        "translateX_9",
        "sharpness_1",

        "sharpness_3",
        "shearY_2",
        "translateY_2",
        "autocontrast_5",
        "equalize_2", #

        "shearY_5",
        "posterize_5", #
        "color_3",
        "brightness_5", #
        "sharpness_9",

        "brightness_9",
        "equalize_5",
        "equalize_1",
        "contrast_7",
        "sharpness_5",
        
        "color_5",
        "translateX_5", #
        "equalize_7",
        "autocontrast_8",
        "translateY_3",
        "sharpness_6",
        "brightness_6",
        "color_8",
        "solarize_0",
        "invert_0", #
        "equalize_0",
        "autocontrast_0", #
        "equalize_8",
        "equalize_4",
        "color_5",
        "equalize_5", #
        "autocontrast_4",
        "solarize_4", #
        "brightness_3",
        "color_0",
        "solarize_1",
        "autocontrast_0", #
        "translateY_3",
        "translateY_3", #
        "autocontrast_1",
        "solarize_1", #
        "equalize_5",
        "invert_0",  #
        "translateY_3",
        "autocontrast_1",
        ]

if __name__ == '__main__':
        print(len(policies))
