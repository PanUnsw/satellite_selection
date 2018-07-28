
def get_concat_iteration_num():
    concat_iteration_nums = {}
    concat_iteration_nums[3] = 7
    concat_iteration_nums[4] = 14
    concat_iteration_nums[5] = 7
    concat_iteration_nums[50] = 7
    concat_iteration_nums[7] = 11
    concat_iteration_nums[8] = 1
    concat_iteration_nums[9] = 5

    concat_iteration_nums[18] = 1
    concat_iteration_nums[19] = 7
    concat_iteration_nums[51] = 7
    concat_iteration_nums[52] = 7
    concat_iteration_nums[522] = 1
    concat_iteration_nums[5222] = 1
    concat_iteration_nums[53] = 7

    return concat_iteration_nums

def get_concat_ids():
    concat_iteration_nums = get_concat_iteration_num()
    concat_ids = {}
    for key,val in concat_iteration_nums.items():
        #concat_ids[key] = [val-1] * val
        concat_ids[key] = [0] * val

    concat_ids[19] = [1, 1, 1, 1, 1, 1, 1]
    concat_ids[50] = [1, 1, 1, 1, 1, 1, 1]

    #concat_ids[50] = [2, 2, 2, 2, 2, 2, 2]
    concat_ids[51] = [1,1,1,1,1,1,1]
    concat_ids[52] = [1, 1, 1, 1, 1, 1, 1]

    concat_ids[522] = [1]
    concat_ids[5222] = [2]
    concat_ids[53] = [1, 1, 1, 1, 1, 1, 1]
    #concat_ids[53] = [2, 2, 2, 2, 2, 2, 2]
    #concat_ids[52] = [1]
    #concat_ids[3][2] = 2

    return concat_ids

def get_config_fconv_multi_concat(flag,num_elements,num_class,num_satelite,k_concat):
    assert flag>=0
        #               kernel_size0,kernel_size1,num_out_channel,stride0,valid_or_same
    config_point_encoders = {}
    config_group_convs = {}
    if flag == 3:

        #config_point_encoders[0] = [[1,num_elements,15,1,0]]
        config_point_encoders[0] = [[1,num_elements,32,1,0],[1,1,32,1,0]]
        #config_point_encoders[2] = [[1,num_elements,50,1,0]]
        config_point_encoders[1] = [[1, num_elements, 64, 1, 0],[1,1,64,1,0]]
        config_point_encoders[2] = [[1,num_elements,128,1,0],[1,1,128,1,0]]
        #config_point_encoders[4] = [[1,num_elements,210,1,0]]
        config_point_encoders[3] = [[1, num_elements, 256, 1, 0],[1,1,256,1,0]]
        #config_point_encoders[6] = [[1, num_elements, 340, 1, 0]]
        config_point_encoders[4] = [[1, num_elements, 512, 1, 0],[1,1,512,1,0]]
        #config_point_encoders[8] = [[1, num_elements, 512, 1, 0]]
        config_point_encoders[5] = [[1, num_elements, 852, 1, 0],[1,1,852,1,0]]
        #config_point_encoders[10] = [[1, num_elements, 700, 1, 0]]
        config_point_encoders[6] = [[1, num_elements, 1024, 1, 0],[1,1,1024,1,0]]

        for i in range(get_concat_iteration_num()[flag]):
            config_group_convs[i] = []

        #               kernel_size0,kernel_size1,num_out_channel,stride0,valid_or_same
        config_classification = [
                                 [1, 1, 1000, 1, 0],

                                 [1, 1, num_class, 1, 0]]
    if flag == 9:

        config_point_encoders[0] = [[1, num_elements,32,1,0]]
        config_point_encoders[1] = [[1, num_elements,104,1,0]]
        config_point_encoders[2] = [[1, num_elements,208,1,0]]
        config_point_encoders[3] = [[1, num_elements, 344, 1, 0]]
        config_point_encoders[4] = [[1, num_elements, 512, 1, 0]]


        for i in range(get_concat_iteration_num()[flag]):
            config_group_convs[i] = []

        #               kernel_size0,kernel_size1,num_out_channel,stride0,valid_or_same
        config_classification = [[1, 1, num_class, 1, 0]]

    if flag == 19:

        config_point_encoders[0] = [[1, num_elements, 32, 1, 0], [1, 1, 35, 1, 0], [1, 1, 50, 1, 0]]
        config_point_encoders[1] = [[1, num_elements, 64, 1, 0], [1, 1, 65, 1, 0], [1, 1, 90, 1, 0]]
        config_point_encoders[2] = [[1, num_elements, 128, 1, 0], [1, 1, 130, 1, 0],
                                    [1, 1, 160, 1, 0]]
        config_point_encoders[3] = [[1, num_elements, 256, 1, 0], [1, 1, 260, 1, 0],
                                    [1, 1, 300, 1, 0]]
        config_point_encoders[4] = [[1, num_elements, 512, 1, 0], [1, 1, 515, 1, 0],
                                    [1, 1, 533, 1, 0]]
        config_point_encoders[5] = [[1, num_elements, 852, 1, 0], [1, 1, 855, 1, 0],
                                    [1, 1, 893, 1, 0]]
        config_point_encoders[6] = [[1, num_elements, 1024, 1, 0], [1, 1, 1025, 1, 0],
                                    [1, 1, 1120, 1, 0]]


        for i in range(get_concat_iteration_num()[flag]):
            config_group_convs[i] = []

        #               kernel_size0,kernel_size1,num_out_channel,stride0,valid_or_same
        config_classification = [[1, 1, 1024, 1, 0],
                                 [1, 1, num_class, 1, 0]]

    if flag == 18:

        config_point_encoders[0] = [[1,num_elements,16,1,0]]

        for i in range(get_concat_iteration_num()[flag]):
            config_group_convs[i] = []

        #               kernel_size0,kernel_size1,num_out_channel,stride0,valid_or_same
        config_classification = [
                                 [1, 1, num_class, 1, 0]]


    if flag == 4:

        config_point_encoders[0] = [[1, num_elements, 32, 1, 0]]
        config_point_encoders[1] = [[1, num_elements, 64, 1, 0]]
        config_point_encoders[2] = [[1, num_elements, 96, 1, 0]]
        config_point_encoders[3] = [[1, num_elements, 128, 1, 0]]
        config_point_encoders[4] = [[1, num_elements, 160, 1, 0]]
        config_point_encoders[5] = [[1, num_elements, 192, 1, 0]]
        config_point_encoders[6] = [[1, num_elements, 224, 1, 0]]
        config_point_encoders[7] = [[1, num_elements, 256, 1, 0]]
        config_point_encoders[8] = [[1, num_elements, 288, 1, 0]]
        config_point_encoders[9] = [[1, num_elements, 320, 1, 0]]
        config_point_encoders[10] = [[1, num_elements, 352, 1, 0]]
        config_point_encoders[11] = [[1, num_elements, 384, 1, 0]]
        config_point_encoders[12] = [[1, num_elements, 416, 1, 0]]
        config_point_encoders[13] = [[1, num_elements, 448, 1, 0]]
        config_point_encoders[14] = [[1, num_elements, 480, 1, 0]]
        config_point_encoders[15] = [[1, num_elements, 512, 1, 0]]

        for i in range(get_concat_iteration_num()[flag]):
            config_group_convs[i] = []

        config_classification = [[1, 1, 896, 1, 0],
                                 [1, 1, 832, 1, 0],
                                 [1, 1, 768, 1, 0],
                                 [1, 1, 704, 1, 0],
                                 [1, 1, 640, 1, 0],
                                 [1, 1, 596, 1, 0],
                                 [1, 1, 612, 1, 0],
                                 [1, 1, 448, 1, 0],
                                 [1, 1, 416, 1, 0],
                                 [1, 1, 384, 1, 0],
                                 [1, 1, 352, 1, 0],
                                 [1, 1, 320, 1, 0],
                                 [1, 1, 288, 1, 0],
                                 [1, 1, 256, 1, 0],
                                 [1, 1, 224, 1, 0],
                                 [1, 1, 192, 1, 0],
                                 [1, 1, 160, 1, 0],
                                 [1, 1, 128, 1, 0],
                                 [1, 1, 96, 1, 0],
                                 [1, 1, 64, 1, 0],
                                 [1, 1, 32, 1, 0],
                                 [1, 1, 16, 1, 0],
                                 [1, 1, num_class, 1, 0]]

    if flag == 5:

        config_point_encoders[0] = [[1,num_elements, 32,1,0],[1,1, 32,1,0]]
        config_point_encoders[1] = [[1,num_elements, 64,1,0],[1,1, 64,1,0]]
        config_point_encoders[2] = [[1,num_elements, 128,1,0],[1,1, 128,1,0]]
        config_point_encoders[3] = [[1, num_elements, 256, 1, 0],[1,1, 256,1,0]]
        config_point_encoders[4] = [[1,num_elements, 512, 1,0],[1,1, 512,1,0]]
        config_point_encoders[5] = [[1, num_elements, 852, 1, 0],[1,1, 852,1,0]]
        config_point_encoders[6] = [[1,num_elements, 1024, 1, 0],[1,1, 1024,1,0]]

        for i in range(get_concat_iteration_num()[flag]):
            config_group_convs[i] = []

        if num_satelite == 16:

            config_group_convs[0] = [[11, 1, 30, 1, 0],
                                     [6, 1, 50, 1, 0],
                                     ]

            config_group_convs[1] = [[11, 1, 60, 1, 0],
                                     [6, 1, 90, 1, 0],
                                     ]

            config_group_convs[2] = [[11, 1, 120, 1, 0],
                                     [6, 1, 160, 1, 0],
                                     ]

            config_group_convs[3] = [[11, 1, 250, 1, 0],
                                     [6, 1, 300, 1, 0],
                                     ]

            config_group_convs[4] = [[11, 1, 480, 1, 0],
                                     [6, 1, 530, 1, 0],
                                     ]

            config_group_convs[5] = [[11, 1, 830, 1, 0],
                                     [6, 1, 890, 1, 0],
                                     ]

            config_group_convs[6] = [[11, 1, 1020, 1, 0],
                                     [6, 1, 1120, 1, 0],
                                     ]




        #               kernel_size0,kernel_size1,num_out_channel,stride0,valid_or_same
        config_classification = [[1, 1, 1024, 1, 0],

                                 [1, 1, num_class, 1, 0]]



    if flag == 55:#  9 model 50
        config_point_encoders[0] = [[1, num_elements, 32, 1, 0], [1, 1, 32, 1, 0],[1, 1, 35, 1, 0],[1, 1, 50, 1, 0]]
        config_point_encoders[1] = [[1, num_elements, 64, 1, 0], [1, 1, 64, 1, 0],[1, 1, 65, 1, 0],[1, 1, 90, 1, 0]]
        config_point_encoders[2] = [[1, num_elements, 128, 1, 0], [1, 1, 128, 1, 0],[1, 1, 130, 1, 0],[1, 1, 160, 1, 0]]
        config_point_encoders[3] = [[1, num_elements, 256, 1, 0], [1, 1, 256, 1, 0],[1, 1, 260, 1, 0],[1, 1, 300, 1, 0]]
        config_point_encoders[4] = [[1, num_elements, 512, 1, 0], [1, 1, 512, 1, 0],[1, 1, 515, 1, 0],[1, 1, 533, 1, 0]]
        config_point_encoders[5] = [[1, num_elements, 852, 1, 0], [1, 1, 852, 1, 0],[1, 1, 855, 1, 0],[1, 1, 893, 1, 0]]
        config_point_encoders[6] = [[1, num_elements, 1024, 1, 0], [1, 1, 1024, 1, 0],[1, 1, 1025, 1, 0],[1, 1, 1120, 1, 0]]

        for i in range(get_concat_iteration_num()[flag]):
            config_group_convs[i] = []

        if num_satelite == 15:

            config_group_convs[0] = [
                                     [16, 1, 50, 1, 0],
                                     ]

            config_group_convs[1] = [
                                     [16, 1, 90, 1, 0],
                                     ]

            config_group_convs[2] = [
                                     [16, 1, 160, 1, 0],
                                     ]

            config_group_convs[3] = [
                                     [16, 1, 303, 1, 0],
                                     ]

            config_group_convs[4] = [
                                     [16, 1, 533, 1, 0],
                                     ]

            config_group_convs[5] = [
                                     [16, 1, 893, 1, 0],
                                     ]

            config_group_convs[6] = [
                                     [16, 1, 1120, 1, 0],
                                     ]

        if num_satelite == 17:

            config_group_convs[0] = [[11, 1, 30, 1, 0],
                                     [6, 1, 50, 1, 0],
                                     ]

            config_group_convs[1] = [[11, 1, 60, 1, 0],
                                     [6, 1, 90, 1, 0],
                                     ]

            config_group_convs[2] = [[11, 1, 120, 1, 0],
                                     [6, 1, 160, 1, 0],
                                     ]

            config_group_convs[3] = [[11, 1, 253, 1, 0],
                                     [6, 1, 303, 1, 0],
                                     ]

            config_group_convs[4] = [[11, 1, 483, 1, 0],
                                     [6, 1, 533, 1, 0],
                                     ]

            config_group_convs[5] = [[11, 1, 833, 1, 0],
                                     [6, 1, 893, 1, 0],
                                     ]

            config_group_convs[6] = [[11, 1, 1023, 1, 0],
                                     [6, 1, 1120, 1, 0],
                                     ]

        #               kernel_size0,kernel_size1,num_out_channel,stride0,valid_or_same
        config_classification = [[1, 1, 1024, 1, 0],

                                 [1, 1, num_class, 1, 0]]


    if flag == 51:

        config_point_encoders[0] = [[1, num_elements, 32, 1, 0],  [1, 1, 35, 1, 0], [1, 1, 50, 1, 0]]
        config_point_encoders[1] = [[1, num_elements, 64, 1, 0],  [1, 1, 65, 1, 0], [1, 1, 90, 1, 0]]
        config_point_encoders[2] = [[1, num_elements, 128, 1, 0],  [1, 1, 130, 1, 0],
                                    [1, 1, 160, 1, 0]]
        config_point_encoders[3] = [[1, num_elements, 256, 1, 0],  [1, 1, 260, 1, 0],
                                    [1, 1, 300, 1, 0]]
        config_point_encoders[4] = [[1, num_elements, 512, 1, 0], [1, 1, 515, 1, 0],
                                    [1, 1, 533, 1, 0]]
        config_point_encoders[5] = [[1, num_elements, 852, 1, 0], [1, 1, 855, 1, 0],
                                    [1, 1, 893, 1, 0]]
        config_point_encoders[6] = [[1, num_elements, 1024, 1, 0], [1, 1, 1025, 1, 0],
                                    [1, 1, 1120, 1, 0]]

        for i in range(get_concat_iteration_num()[flag]):
            config_group_convs[i] = []



        #               kernel_size0,kernel_size1,num_out_channel,stride0,valid_or_same
        config_classification = [[1, 1, 1024, 1, 0],

                                 [1, 1, num_class, 1, 0]]

    if flag == 52:

        config_point_encoders[0] = [[1, num_elements, 32, 1, 0], [1, 1, 32, 1, 0], [1, 1, 48, 1, 0]]
        config_point_encoders[1] = [[1, num_elements, 64, 1, 0], [1, 1, 64, 1, 0], [1, 1, 96, 1, 0]]
        config_point_encoders[2] = [[1, num_elements, 128, 1, 0], [1, 1, 128, 1, 0],
                                    [1, 1, 160, 1, 0]]
        config_point_encoders[3] = [[1, num_elements, 256, 1, 0], [1, 1, 256, 1, 0],
                                    [1, 1, 320, 1, 0]]
        config_point_encoders[4] = [[1, num_elements, 512, 1, 0], [1, 1, 512, 1, 0],
                                    [1, 1, 640, 1, 0]]
        config_point_encoders[5] = [[1, num_elements, 864, 1, 0], [1, 1, 864, 1, 0],
                                    [1, 1, 960, 1, 0]]
        config_point_encoders[6] = [[1, num_elements, 1024, 1, 0], [1, 1, 1024, 1, 0],
                                     [1, 1, 1152, 1, 0]]

        for i in range(get_concat_iteration_num()[flag]):
            config_group_convs[i] = []

        #               kernel_size0,kernel_size1,num_out_channel,stride0,valid_or_same
        config_classification = [[1, 1, 1024, 1, 0],

                                 [1, 1, num_class, 1, 0]]

    if flag == 522:

        config_point_encoders[0] = [[1, num_elements, 32, 1, 0],
                                    [1, 1, 64, 1, 0],
                                    [1, 1, 128, 1, 0],
                                    [1, 1, 256, 1, 0],
                                    [1, 1, 512, 1, 0],
                                    [1, 1, 1024, 1, 0]
                                    ]


        for i in range(get_concat_iteration_num()[flag]):
            config_group_convs[i] = []

        #               kernel_size0,kernel_size1,num_out_channel,stride0,valid_or_same
        config_classification = [
                                 [1, 1, 512, 1, 0],
                                 [1, 1, num_class, 1, 0]]

    if flag == 5222:

        config_point_encoders[0] = [[1, num_elements, 32, 1, 0],[1, 1, 32, 1, 0], [1, 1, 48, 1, 0],
                                    [1, 1, 64, 1, 0],[1, 1, 64, 1, 0], [1, 1, 96, 1, 0],
                                    [1, 1, 128, 1, 0],[1, 1, 128, 1, 0],
                                    [1, 1, 160, 1, 0],
                                    [1, 1, 256, 1, 0],[1, 1, 256, 1, 0],
                                    [1, 1, 320, 1, 0],
                                    [1, 1, 512, 1, 0],[1, 1, 512, 1, 0],
                                    [1, 1, 640, 1, 0],[1, 1, 864, 1, 0],[1, 1, 864, 1, 0],
                                    [1, 1, 960, 1, 0],
                                    [1, 1, 1024, 1, 0],[1, 1, 1024, 1, 0],
                                     [1, 1, 1152, 1, 0]
                                    ]

        for i in range(get_concat_iteration_num()[flag]):
            config_group_convs[i] = []

        #               kernel_size0,kernel_size1,num_out_channel,stride0,valid_or_same
        config_classification = [[1, 1, 1024, 1, 0],
                                 [1, 1, 512, 1, 0],
                                 [1, 1, num_class, 1, 0]]


    if flag == 53:

        config_point_encoders[0] = [[1, num_elements, 32, 1, 0], [1, 1, 32, 1, 0], [1, 1, 48, 1, 0]]
        config_point_encoders[1] = [[1, num_elements, 64, 1, 0], [1, 1, 64, 1, 0], [1, 1, 96, 1, 0]]
        config_point_encoders[2] = [[1, num_elements, 128, 1, 0], [1, 1, 128, 1, 0],
                                    [1, 1, 160, 1, 0]]
        config_point_encoders[3] = [[1, num_elements, 256, 1, 0], [1, 1, 256, 1, 0],
                                    [1, 1, 352, 1, 0]]
        config_point_encoders[4] = [[1, num_elements, 512, 1, 0], [1, 1, 512, 1, 0],
                                    [1, 1, 672, 1, 0]]
        config_point_encoders[5] = [[1, num_elements, 864, 1, 0], [1, 1, 864, 1, 0],
                                    [1, 1, 992, 1, 0]]
        config_point_encoders[6] = [[1, num_elements, 1024, 1, 0], [1, 1, 1024, 1, 0],
                                    [1, 1, 1184, 1, 0]]


       # config_point_encoders[0] = [[1, num_elements, 32, 1, 0], [1, 1, 32, 1, 0], [1, 1, 48, 1, 0]]
       # config_point_encoders[1] = [[1, num_elements, 64, 1, 0], [1, 1, 64, 1, 0], [1, 1, 96, 1, 0]]
       # config_point_encoders[2] = [[1, num_elements, 128, 1, 0], [1, 1, 128, 1, 0],
                                   # [1, 1, 160, 1, 0]]
        #config_point_encoders[3] = [[1, num_elements, 256, 1, 0], [1, 1, 256, 1, 0],
                                   # [1, 1, 352, 1, 0]]
       # config_point_encoders[4] = [[1, num_elements, 512, 1, 0], [1, 1, 512, 1, 0],
                                   # [1, 1, 672, 1, 0]]
       # config_point_encoders[5] = [[1, num_elements, 864, 1, 0], [1, 1, 864, 1, 0],
                                   # [1, 1, 992, 1, 0]]
       # config_point_encoders[6] = [[1, num_elements, 1024, 1, 0], [1, 1, 1024, 1, 0],
                                   # [1, 1, 1184, 1, 0]]



        for i in range(get_concat_iteration_num()[flag]):
            config_group_convs[i] = []

        #               kernel_size0,kernel_size1,num_out_channel,stride0,valid_or_same
        config_classification = [[1, 1, 1024, 1, 0],

                                 [1, 1, num_class, 1, 0]]

    if flag == 50:#12 model 50

        config_point_encoders[0] = [[1, num_elements, 32, 1, 0], [1, 1, 32, 1, 0], [1, 1, 48, 1, 0]]
        config_point_encoders[1] = [[1, num_elements, 64, 1, 0], [1, 1, 64, 1, 0], [1, 1, 96, 1, 0]]
        config_point_encoders[2] = [[1, num_elements, 128, 1, 0], [1, 1, 128, 1, 0],
                                    [1, 1, 160, 1, 0]]
        config_point_encoders[3] = [[1, num_elements, 256, 1, 0], [1, 1, 256, 1, 0],
                                    [1, 1, 320, 1, 0]]
        config_point_encoders[4] = [[1, num_elements, 512, 1, 0], [1, 1, 512, 1, 0],
                                    [1, 1, 640, 1, 0]]
        config_point_encoders[5] = [[1, num_elements, 864, 1, 0], [1, 1, 864, 1, 0],
                                    [1, 1, 960, 1, 0]]
        config_point_encoders[6] = [[1, num_elements, 1024, 1, 0], [1, 1, 1024, 1, 0],
                                    [1, 1, 1152, 1, 0]]

        for i in range(get_concat_iteration_num()[flag]):
            config_group_convs[i] = []


        #               kernel_size0,kernel_size1,num_out_channel,stride0,valid_or_same
        config_classification = [[1, 1, 1024, 1, 0],
                                 [1, 1, num_class, 1, 0]]

    if flag == 6:

        config_point_encoders[0] = [[1, num_elements, 32, 1, 0], [1, 1, 32, 1, 0]]
        config_point_encoders[1] = [[1, num_elements, 64, 1, 0], [1, 1, 64, 1, 0]]
        config_point_encoders[2] = [[1, num_elements, 96, 1, 0], [1, 1, 96, 1, 0]]
        config_point_encoders[3] = [[1, num_elements, 128, 1, 0], [1, 1, 128, 1, 0]]
        config_point_encoders[4] = [[1, num_elements, 160, 1, 0], [1, 1, 160, 1, 0]]
        config_point_encoders[5] = [[1, num_elements, 192, 1, 0], [1, 1, 192, 1, 0]]
        config_point_encoders[6] = [[1, num_elements, 224, 1, 0], [1, 1, 224, 1, 0]]
        config_point_encoders[7] = [[1, num_elements, 256, 1, 0], [1, 1, 256, 1, 0]]
        config_point_encoders[8] = [[1, num_elements, 288, 1, 0], [1, 1, 288, 1, 0]]
        config_point_encoders[9] = [[1, num_elements, 320, 1, 0], [1, 1, 320, 1, 0]]
        config_point_encoders[10] = [[1, num_elements, 352, 1, 0], [1, 1, 352, 1, 0]]
        config_point_encoders[11] = [[1, num_elements, 384, 1, 0], [1, 1, 384, 1, 0]]
        config_point_encoders[12] = [[1, num_elements, 416, 1, 0], [1, 1, 416, 1, 0]]
        config_point_encoders[13] = [[1, num_elements, 448, 1, 0], [1, 1, 448, 1, 0]]
        config_point_encoders[14] = [[1, num_elements, 480, 1, 0], [1, 1, 480, 1, 0]]
        config_point_encoders[15] = [[1, num_elements, 512, 1, 0], [1, 1, 512, 1, 0]]

        for i in range(get_concat_iteration_num()[flag]):
            config_group_convs[i] = []

        config_classification = [
                                [1, 1, 512, 1, 0],
                                [1, 1, 480, 1, 0],
                                [1, 1, 448, 1, 0],
                                [1, 1, 416, 1, 0],
                                [1, 1, 384, 1, 0],
                                [1, 1, 352, 1, 0],
                                [1, 1, 320, 1, 0],
                                [1, 1, 288, 1, 0],
                                [1, 1, 256, 1, 0],
                                [1, 1, 224, 1, 0],
                                [1, 1, 192, 1, 0],
                                [1, 1, 160, 1, 0],
                                [1, 1, 128, 1, 0],
                                [1, 1, 96, 1, 0],
                                [1, 1, 64, 1, 0],
                                [1, 1, 32, 1, 0],
                                [1, 1, 16, 1, 0],
                                [1, 1, num_class, 1, 0]]
    if flag == 7:

        config_point_encoders[0] = [[1,num_elements,32,1,0]]

        config_point_encoders[1] = [[1,num_elements,104,1,0]]

        config_point_encoders[2] = [[1, num_elements, 152, 1, 0]]

        config_point_encoders[3] = [[1,num_elements,208,1,0]]

        config_point_encoders[4] = [[1, num_elements, 344, 1, 0]]

        config_point_encoders[5] = [[1, num_elements, 512, 1, 0]]

        config_point_encoders[6] = [[1, num_elements, 712, 1, 0]]

        config_point_encoders[7] = [[1, num_elements, 944, 1, 0]]

        config_point_encoders[8] = [[1, num_elements, 1208, 1, 0]]

        config_point_encoders[9] = [[1, num_elements, 1504, 1, 0]]

        #config_point_encoders[10] = [[1, num_elements, 1832, 1, 0]]


        for i in range(get_concat_iteration_num()[flag]):
            config_group_convs[i] = []

        #               kernel_size0,kernel_size1,num_out_channel,stride0,valid_or_same
        config_classification = [[1, 1, 1832, 1, 0],
                                 [1, 1, 1504, 1, 0],
                                 [1, 1, 1072, 1, 0],
                                 [1, 1, 944, 1, 0],
                                 [1, 1, 824, 1, 0],
                                 [1, 1, 712, 1, 0],
                                 [1, 1, 608, 1, 0],
                                 [1, 1, 512, 1, 0],
                                 [1, 1, 424, 1, 0],
                                 [1, 1, 344, 1, 0],
                                 [1, 1, 272, 1, 0],
                                 [1, 1, 208, 1, 0],
                                 [1, 1, 152, 1, 0],
                                 [1, 1, 104, 1, 0],
                                 [1, 1, 64, 1, 0],
                                 [1, 1, 32, 1, 0],
                                 [1, 1, 16, 1, 0],
                                 [1, 1, num_class, 1, 0]]
    if flag == 8:

        config_point_encoders[0] = [[1,num_elements,32,1,0], [1,1,32,1,0]]

        for i in range(get_concat_iteration_num()[flag]):
            config_group_convs[i] = []

        #               kernel_size0,kernel_size1,num_out_channel,stride0,valid_or_same
        config_classification = [
                                 [1, 1, num_class, 1, 0]]



    config_point_encoder = config_point_encoders[k_concat]
    config_group_conv = config_group_convs[k_concat]
    concat_ids = get_concat_ids()[flag]

    return config_point_encoder, config_group_conv, config_classification, concat_ids

