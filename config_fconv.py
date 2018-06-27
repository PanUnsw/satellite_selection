
def get_config_fconv(flag,num_elements,num_class,num_satelite):
    assert flag>=0
    #***************************************************************************
    # flag < 50: use maxpooling as group encoder:  config_group_conv = []
    config_group_conv = []
    if flag == 1:
        config_point_encoder = [[1,num_elements,32,1,0]]
        #               kernel_size0,kernel_size1,num_out_channel,stride0,valid_or_same
        config_classification = [
                                 [1,1,num_class,1,0]]
    if flag == 5:
        config_point_encoder = [[1,num_elements,32,1,0],
                                [1,1,32,1,0],
                                [1,1,64,1,0],
                                [1,1,128,1,0]]
        #               kernel_size0,kernel_size1,num_out_channel,stride0,valid_or_same
        config_classification = [[1,1,256,1,0],
                                 [1,1,128,1,0],
                                 [1,1,64,1,0],
                                 [1,1,num_class,1,0]]
    if flag == 6:
        config_point_encoder = [[1,num_elements,32,1,0],
                                [1,1,64,1,0],
                                [1,1,128,1,0]]
        #               kernel_size0,kernel_size1,num_out_channel,stride0,valid_or_same
        config_classification = [[1,1,256,1,0],
                                 [1,1,64,1,0],
                                 [1,1,num_class,1,0]]



    #***************************************************************************
    # flag > 50: use fully convolution as group encoder
        #               kernel_size0,kernel_size1,num_out_channel,stride0,valid(0)_or_same(1)
    if flag == 98:
    #                  [kernel_size0,kernel_size1,num_out_channel]
        config_point_encoder = [[1,num_elements,24,1,0]]
        #               kernel_size0,kernel_size1,num_out_channel,stride0,valid_or_same
        config_group_conv = [[56,1,48,56,1]]
        config_classification = [[1,1,24,1,0],
                        [1,1,num_class,1,0]]
    if flag == 99:
    #                  [kernel_size0,kernel_size1,num_out_channel]
        config_point_encoder = [[1,num_elements,24,1,0]]
        #               kernel_size0,kernel_size1,num_out_channel,stride0,valid_or_same
        config_group_conv = [[20,1,32,9,1],
                        [16,1,48,7,1]]
        config_classification = [[1,1,32,1,0],
                        [1,1,num_class,1,0]]
    if flag == 100:
    #                  [kernel_size0,kernel_size1,num_out_channel]
        config_point_encoder = [[1,num_elements,24,1,0]]
        #               kernel_size0,kernel_size1,num_out_channel,stride0,valid_or_same
        config_group_conv = [[11,1,32,3,0],
                             [8,1,48,4,0],
                             [3,1,64,1,0]]
        config_classification = [[1,1,32,1,0],
                                 [1,1,num_class,1,0]]
    if flag == 101:
        config_point_encoder = [[1,num_elements,32,1,0],
                                [1, 1, 64,  1, 0],
                                [1, 1, 128, 1, 0],
                                [1, 1, 256, 1, 0],
                                 ]
        #               kernel_size0,kernel_size1,num_out_channel,stride0,valid_or_same
        #config_point_encoder = [[1, num_elements, 32, 1, 0],
         #                       [1, 1, 64, 1, 0]]
        if num_satelite==56:
             config_group_conv = [[14,1,128,2,0],
                                  [10,1,256,1,0],
                                  [7,1,512,2,0],
                                  [4, 1, 1024, 1, 0]]

            #config_group_conv = [[24,1,64,2,0],
             #                    [17,1,128,1,0]]


            #config_group_conv = [[11, 1, 32, 3, 0],
              #                   [8, 1, 48, 4, 0],
             #                    [3, 1, 64, 1, 0]]

            #config_group_conv = [[16, 1, 128, 2, 0],
             #                    [13, 1, 256, 1, 0],
              #                   [7, 1, 512, 1, 0],
              #                   [3, 1, 512, 1, 0]]

            #config_group_conv = [[15, 1, 128, 1, 0],
                             #    [12, 1, 256, 2, 0],
                              #   [10, 1, 512, 2, 0],
                             #    [4, 1, 512, 1, 0]]

        elif num_satelite == 25:
            config_group_conv = [[9,1,128,1,0],
                                 [9,1,256,2,0],
                                 [5,1,512,1,0]]
        elif num_satelite == 18:
            config_group_conv = [[11, 1, 128, 1, 0],
                                 [6, 1, 256, 1, 0],
                                 [3, 1, 512, 1, 0],
                                 [1, 1, 1024, 1, 0]]
        elif num_satelite == 17:
            config_group_conv = [[9, 1, 128, 1, 0],
                                 [7, 1, 256, 1, 0],
                                 [3, 1, 512, 1, 0],
                                 [1, 1, 1024, 1, 0]]
        elif num_satelite == 16:
           config_group_conv = [[9,1,128,1,0],
                                [6,1,256,1,0],
                                [3,1,512,1,0],
                                [1, 1, 1024, 1, 0]]
        elif num_satelite == 15:
           config_group_conv = [[8,1,128,1,0],
                                [5,1,256,1,0],
                                [4,1,512,1,0],
                                [1, 1, 1024, 1, 0]]

        elif num_satelite == 13:
            config_group_conv = [[7,1,128,1,0],
                                 [5,1,256,1,0],
                                 [3, 1, 512, 1, 0],
                                 [1, 1, 1024, 1, 0]]

        elif num_satelite == 12:
            config_group_conv = [[6,1,128,1,0],
                                 [4,1,256,1,0],
                                 [4, 1, 512, 1, 0],
                                 [1, 1, 1024, 1, 0]
                                 ]

        elif num_satelite == 11:
            config_group_conv = [[10,1,128,1,0],
                                 [2,1,256,1,0],
                                 [1, 1, 512, 1, 0]]

        elif num_satelite == 10:
            config_group_conv = [[6, 1, 128, 1, 0],
                                 [4, 1, 256, 1, 0],
                                 [2, 1, 512, 1, 0]]

            #config_group_conv = [[9,1,128,1,0],
             #                    [2,1,256,1,0]]

        elif num_satelite == 9:
            config_group_conv = [[9,1,128,1,0],
                                 [1, 1, 256, 1, 0],
                                 [1, 1, 512, 1, 0]
                                 ]

        config_classification = [[1,1,512,1,0],
                                 [1,1,256,1,0],
                                 [1,1,128,1,0],
                                 [1, 1, 64, 1, 0],
                                 [1, 1, 32, 1, 0],
                                 [1,1,num_class,1,0]]

        #config_classification = [[1, 1, 64, 1, 0],
          #                       [1,1,num_class,1,0]]

    if flag == 102:
        config_point_encoder = [[1, 1, 128, 1, 0],
                                [1, 1, 128, 1, 0]]
        #               kernel_size0,kernel_size1,num_out_channel,stride0,valid_or_same
        config_group_conv = [[12, 1, 128, 2, 0],
                             [11, 1, 256, 2, 0],
                             [7, 1, 512, 1, 0]]

        config_classification = [[1, 1, 256, 1, 0],
                                 [1, 1, 64, 1, 0],
                                 [1, num_elements, num_class, 1, 0]]

    return config_point_encoder, config_group_conv, config_classification
