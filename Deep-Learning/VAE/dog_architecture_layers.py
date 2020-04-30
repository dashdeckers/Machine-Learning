# https://medium.com/@claymason313/dog-breed-image-classification-1ef7dc1b1967
# Encoder:
self.layer_list = [
            # Dog breed image classification from: 
            # https://medium.com/@claymason313/dog-breed-image-classification-1ef7dc1b1967
            layers.Conv2D(input_shape[1], (3, 3), input_shape=input_shape, activation='relu', data_format='channels_last'),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(input_shape[1], (3,3), activation='relu', data_format='channels_last'),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(input_shape[1]*2, (3, 3), activation='relu', data_format='channels_last'),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dense(input_shape[1]*2, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(self.latent_dim, activation='softmax')
        ]

# Decoder:
self.layer_list = [
            layers.Dense(self.latent_dim, activation='softmax'),  # (120)
            layers.Dense(self.latent_dim / 2, activation='relu'),  # (64)
            layers.Dense(self.latent_dim / 2, activation='relu'),  # (64,)
            layers.Dense(256, activation='relu'),  # 256
            layers.Reshape((2, 2, -1)),  # (2,2,64)
            layers.UpSampling2D(size=(2, 2)),  # (4, 4, 64)
            layers.Conv2DTranspose(self.im_dim[0], (3, 3), activation='relu', data_format='channels_last'),  # 6, 6, 32)
            layers.UpSampling2D(size=(2, 2)),  # (12,12,32)
            layers.Conv2DTranspose(self.im_dim[0], (4, 4), activation='relu', data_format='channels_last'),  # (15,15,32)
            layers.UpSampling2D(size=(2, 2)),  # (30,30,32)
            layers.Conv2DTranspose(self.im_dim[2], (3, 3), activation='relu', data_format='channels_last'),  # (32, 32, 3)
        ]

#Travis branch
exp['encoder_layers'] = [       # loss = binary crossentropy
        # Dog breed image classification from:
        # https://medium.com/@claymason313/dog-breed-image-classification-1ef7dc1b1967
        layers.Conv2D(
            exp['input_shape'][1], (3, 3), input_shape=exp['input_shape'],
            activation='relu', data_format='channels_last'
            ),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(
            exp['input_shape'][1], (3, 3), activation='relu',
            data_format='channels_last'
            ),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(
            exp['input_shape'][1]*2, (3, 3), activation='relu',
            data_format='channels_last'
            ),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(exp['input_shape'][1]*2, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(exp['latent_dim'], activation='softmax')
    ]

exp['decoder_layers'] = [
        layers.Dense(exp['latent_dim'], activation='softmax'),  # (120)
        layers.Dense(exp['latent_dim'] / 2, activation='relu'),  # (64)
        layers.Dense(exp['latent_dim'] / 2, activation='relu'),  # (64,)
        layers.Dense(256, activation='relu'),  # 256
        layers.Reshape((2, 2, -1)),  # (2,2,64)
        layers.UpSampling2D(size=(2, 2)),  # (4, 4, 64)
        layers.Conv2DTranspose(
            exp['im_shape'][0], (3, 3), activation='relu',
            data_format='channels_last'
            ),  # 6, 6, 32)
        layers.UpSampling2D(size=(2, 2)),  # (12,12,32)
        layers.Conv2DTranspose(
            exp['im_shape'][0], (4, 4), activation='relu',
            data_format='channels_last'
            ),  # (15,15,32)
        layers.UpSampling2D(size=(2, 2)),  # (30,30,32)
        layers.Conv2DTranspose(
            exp['channels'], (3, 3), activation='relu',
            data_format='channels_last'
            ),  # (32, 32, 3)
        layers.Reshape(target_shape=exp['input_shape'][1:]),
    ]