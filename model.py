from typing import Optional
import torch
from torch import nn
# Local imports
from perceiverIO.decoder import BaseDecoder, PerceiverDecoder
from perceiverIO.encoder import PerceiverEncoder
from perceiverIO.positional_encoding import PositionalEncodingPermute3D as pe3d
from perceiverIO.summer import Summer


class PerceiverIO(nn.Module):
    """
    Perceiver IO encoder-decoder architecture.
    """
    def __init__(self,
                 input_data_shape,
                 num_latents=12,   # Number of latent vectors to output
                 latent_dim=512,   # Dimensionality of each vector
                 n_heads=4,
                 encoder: Optional[PerceiverEncoder] = None,
                 decoder: Optional[BaseDecoder] = None,
                 q_name: Optional[str] = None,
                 positional_encode=True
                 ):
        """Constructor.
        Args:
            encoder: Instance of Perceiver IO encoder.
            decoder: Instance of Perceiver IO decoder.

            Everything is magic numbers atm :eyes: .
        """
        super().__init__()
        self.batch_size, d_time, d_image, d_i = input_data_shape
        self.num_latents = num_latents
        self.latent_dim = latent_dim
        self.n_heads = n_heads

        # Test for shapes
        test_data = torch.rand(input_data_shape)
        test_data = test_data.unsqueeze(1)

        self.pos_sum = Summer(pe3d(4))

        # if no enc or dec given make them follow the shapes.
        if encoder is None:
            self.encoder = PerceiverEncoder(
                            num_latents=self.num_latents,
                            latent_dim=self.latent_dim,
                            input_dim=input_data_shape,
                )
        if decoder is None:
            self.decoder = PerceiverDecoder(
                            num_latents=self.num_latents,
                            latent_dim=self.latent_dim,
                            n_heads=self.n_heads
                )
        self.qf = self.select_qf(q_name)

    def forward(
        self,
        inputs: torch.Tensor,
        query: Optional[torch.Tensor] = None,
        input_mask: Optional[torch.Tensor] = None,
        query_mask: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            inputs: Input tensor.
            query: Decoder query tensor. Can be a trainable or hand-made.
                Defaults to None.
            input_mask: Input mask tensor. Mask values selected in [0, 1].
                Defaults to None.
            query_mask: Decoder query mask tensor. Mask values selected in
                [0, 1]. Defaults to None.
        Returns:
            Output tensor.
        """

        bs = inputs.shape[0]
        inputs = inputs.unsqueeze(1)
        conved = nn.functional.relu(self.cv1(inputs))
        conved = self.maxp(conved)
        conved = self.pos_sum(conved)
        conved = conved.flatten(1).unsqueeze(1)
        latents = self.encoder(conved)
        if query is None:
            query = self.qf(bs, inputs)
        outputs = self.decoder(
            query=query,
            latents=latents
        )
        return outputs.reshape(-1, 24, 64, 64)

    def select_qf(self, q_name):
        """
        Select query function, and perform any necessary inits for the model to
        have such a qf.
        Conv2 performs very slightly better than maxpool, but do not think it
        is worth the size.
        TODO: Should write an explanation on how this works
        """
        if q_name == "conv1":
            return self.conv_query
        if q_name == "conv2":
            self.cvQ = nn.Conv3d(1,
                                 2,
                                 (1, 2, 2),
                                 stride=(1, 2, 2))
            return self.conv_query2
        elif q_name == "maxpool":
            self.mp1 = nn.MaxPool3d((1, 2, 2), return_indices=True)
            return self.maxpool_query
        else:
            raise Exception

    def conv_query(self, bs, inputs):
        """
        Produce a query based on the conv layer output:
            - Benefits:
                ¬ Maybe better trains the conv layer
                ¬ Less parameters
            - Disadvantages:
                ¬ Maybe less specialisation is better
                ¬ Requires more operations
                ¬ Not using it allows more freedom with conv layer size
        """
        conved = self.cv1(inputs)
        conved = nn.functional.relu(self.cv1(inputs))
        return conved.reshape(bs, 24, -1)

    def conv_query2(self, bs, inputs):
        """
        Produce a query using it's own conv layer!
        """
        conved = self.cvQ(inputs)
        conved = nn.functional.relu(conved)
        return conved.reshape(bs, 24, 4096)

    def maxpool_query(self, bs, inputs):
        """
        MaxPool the 2 highest numbers from each 2x2 section so:
            [12,128,128] ---> [12,2,64,64] / [24,64,64]
        Actually think might be better to conc together the max pool along with
        the indices!
        """
        vals, indices = self.mp1(inputs)
        query = torch.cat((vals, indices), axis=1)
        return query.reshape(bs, 24, 4096)


if __name__ == '__main__':
    batch_data_shape = (30, 12, 128, 128)
    target_data_shape = (30, 24, 64, 64)
    coords_data_shape = (30, 2, 128, 128)
    test_data = torch.rand(batch_data_shape).to("cuda")

    percy = PerceiverIO(test_data).to("cuda")
    n_parameters = sum([param.nelement() for param in percy.parameters()])
    print("Number of parameters:", n_parameters)
    test_forward = percy(test_data)
    print("Forward:\t", test_forward.shape)

    # Check size
    import os
    print("saving")
    torch.save(percy.state_dict(), './model.pt')
    file_size = os.path.getsize("./model.pt")
    print("Model Size:\t", file_size / (1024*1024.0), "MB")
