import json
import os
import nugraph as ng
import torch
import numpy as np
import pandas as pd
import torch_geometric as pyg

from typing import Callable
from torch import nn
from torch_geometric.transforms import Compose
from torch_geometric.data import Dataset
# triton_python_backend_utils is available in every Triton Python model. You
# need to use this module to create inference requests and responses. It also
# contains some utility functions for extracting information from model_config
# and converting Triton input/output types to numpy types.
import triton_python_backend_utils as pb_utils

class HitGraphProducer():
    def __init__(self,
                 semantic_labeller: Callable = None,
                 event_labeller: Callable = None,
                 label_vertex: bool = False,
                 planes: list[str] = ['u','v','y'],
                 node_pos: list[str] = ['local_wire','local_time'],
                 pos_norm: list[float] = [0.3,0.055],
                 node_feats: list[str] = ['integral','rms'],
                 lower_bound: int = 20,
                 filter_hits: bool = False):

        self.semantic_labeller = semantic_labeller
        self.event_labeller = event_labeller
        self.label_vertex = label_vertex
        self.planes = planes
        self.node_pos = node_pos
        self.pos_norm = torch.tensor(pos_norm).float()
        self.node_feats = node_feats
        self.lower_bound = lower_bound
        self.filter_hits = filter_hits

        self.transform = pyg.transforms.Compose((
            pyg.transforms.Delaunay(),
            pyg.transforms.FaceToEdge()))
    
    def create_graph(self, hit_table_hit_id, hit_table_local_plane, hit_table_local_time, \
                    hit_table_local_wire, hit_table_integral, hit_table_rms, \
                    spacepoint_table_spacepoint_id, spacepoint_table_hit_id_u, spacepoint_table_hit_id_v, \
                    spacepoint_table_hit_id_y):
        evt = {
            'hit_table': pd.DataFrame({
                'hit_id':hit_table_hit_id, 'local_plane': hit_table_local_plane, 'local_time':hit_table_local_time, \
                'local_wire':hit_table_local_wire, 'integral':hit_table_integral, 'rms':hit_table_rms
            }),
            'spacepoint_table': pd.DataFrame({
                'spacepoint_id':spacepoint_table_spacepoint_id, 'hit_id_u':spacepoint_table_hit_id_u, \
                'hit_id_v':spacepoint_table_hit_id_v, 'hit_id_y':spacepoint_table_hit_id_y
            })
                                
        }
        if self.event_labeller or self.label_vertex:
            event = evt['event_table'].squeeze()

        hits = evt['hit_table']
        spacepoints = evt['spacepoint_table'].reset_index(drop=True)

        # discard any events with pathologically large hit integrals
        # this is a hotfix that should be removed once the dataset is fixed
        if hits.integral.max() > 1e6:
            print('found event with pathologically large hit integral, skipping')
            return evt.name, None

        # handle energy depositions
        if self.filter_hits or self.semantic_labeller:
            edeps = evt['edep_table']
            energy_col = 'energy' if 'energy' in edeps.columns else 'energy_fraction' # for backwards compatibility
            edeps = edeps.sort_values(by=[energy_col],
                                      ascending=False,
                                      kind='mergesort').drop_duplicates('hit_id')
            hits = edeps.merge(hits, on='hit_id', how='right')

            # if we're filtering out data hits, do that
            if self.filter_hits:
                hitmask = hits[energy_col].isnull()
                filtered_hits = hits[hitmask].hit_id.tolist()
                hits = hits[~hitmask].reset_index(drop=True)
                # filter spacepoints from noise
                cols = [ f'hit_id_{p}' for p in self.planes ]
                spmask = spacepoints[cols].isin(filtered_hits).any(axis='columns')
                spacepoints = spacepoints[~spmask].reset_index(drop=True)

            hits['filter_label'] = ~hits[energy_col].isnull()
            hits = hits.drop(energy_col, axis='columns')

        # reset spacepoint index
        spacepoints = spacepoints.reset_index(names='index_3d')

        # get labels for each particle
        if self.semantic_labeller:
            particles = self.semantic_labeller(evt['particle_table'])
            try:
                hits = hits.merge(particles, on='g4_id', how='left')
            except:
                print('exception occurred when merging hits and particles')
                print('hit table:', hits)
                print('particle table:', particles)
                print('skipping this event')
                return None
            mask = (~hits.g4_id.isnull()) & (hits.semantic_label.isnull())
            if mask.any():
                print(f'found {mask.sum()} orphaned hits.')
                return evt.name, None
            del mask

        data = pyg.data.HeteroData()

        # event metadata
        data['metadata'].run = 6876
        data['metadata'].subrun = 9
        data['metadata'].event = 470

        # spacepoint nodes
        data['sp'].num_nodes = spacepoints.shape[0]

        # draw graph edges
        for i, plane_hits in hits.groupby('local_plane'):

            p = self.planes[i]
            plane_hits = plane_hits.reset_index(drop=True).reset_index(names='index_2d')

            # node position
            pos = torch.tensor(plane_hits[self.node_pos].values).float()
            data[p].pos = pos * self.pos_norm[None,:]

            # node features
            data[p].x = torch.tensor(plane_hits[self.node_feats].values).float()

            # hit indices
            data[p].id = torch.tensor(plane_hits['hit_id'].values).long()

            # 2D edges
            data[p, 'plane', p].edge_index = self.transform(data[p]).edge_index

            # 3D edges
            edge3d = spacepoints.merge(plane_hits[['hit_id','index_2d']].add_suffix(f'_{p}'),
                                       on=f'hit_id_{p}',
                                       how='inner')
            edge3d = edge3d[[f'index_2d_{p}','index_3d']].values.transpose()
            edge3d = torch.tensor(edge3d) if edge3d.size else torch.empty((2,0))
            data[p, 'nexus', 'sp'].edge_index = edge3d.long()

            # truth information
            if self.semantic_labeller:
                data[p].y_semantic = torch.tensor(plane_hits['semantic_label'].fillna(-1).values).long()
                data[p].y_instance = torch.tensor(plane_hits['instance_label'].fillna(-1).values).long()
            if self.label_vertex:
                vtx_2d = torch.tensor([ event[f'nu_vtx_wire_pos_{i}'], event.nu_vtx_wire_time ]).float()
                data[p].y_vtx = vtx_2d * self.pos_norm[None,:]

        for p in self.planes:
            if bool(data[p]): continue
            data[p].pos = torch.empty(0, 2)
            data[p].x = torch.empty(0, 2)
            data[p].id = torch.empty(0)
            data[p, 'plane', p].edge_index = torch.empty((2, 0), dtype=torch.long)
            data[p, 'nexus', 'sp'].edge_index = torch.empty((2, 0), dtype=torch.long)

        # event label
        if self.event_labeller:
            data['evt'].y = torch.tensor(self.event_labeller(event)).long()

        # 3D vertex truth
        if self.label_vertex:
            vtx_3d = [ [ event.nu_vtx_corr_x, event.nu_vtx_corr_y, event.nu_vtx_corr_z ] ]
            data['evt'].y_vtx = torch.tensor(vtx_3d).float()
        
        return data
    
class HeteroDataset(Dataset):
    def __init__(self, hetero_data, transform=None):
        super().__init__(transform=transform)
        self.transform = transform
        self.hetero_data = hetero_data

    def len(self):
        return 1
    def get(self, idx=0):
        return self.transform(self.hetero_data)
    
class NuGraph2_model(nn.Module):
    """
    Simple AddSub network in PyTorch. This network outpxuts the sum and
    subtraction of the inputs.
    """

    def __init__(self):
        super(NuGraph2_model, self).__init__()
        self.MODEL = ng.models.nugraph2.NuGraph2
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        modelpath = os.path.dirname(os.path.abspath(__file__))
        self.model = self.MODEL.load_from_checkpoint(os.path.join(modelpath, "test-ng2.ckpt"), map_location=self.device)
        self.planes = ['u', 'v', 'y']
        self.norm = {'u':torch.tensor(np.array([[168.04594, 178.3245 , 266.6149 , 3.857218 ], [ 82.80644, 67.60649 , 274.32666, 1.2912455 ]]).astype(np.float32)),
                     'v':torch.tensor(np.array([[1245.3547, 176.54117, 323.52786, 4.3267984], [293.06314, 66.8194  , 322.11386,  1.4249923]]).astype(np.float32)),
                     'y':torch.tensor(np.array([[1225.5012, 183.58075, 310.83493, 4.3409133], [307.1943 , 67.063324, 312.461  ,  1.4532351]]).astype(np.float32))}

        self.hitgraph = HitGraphProducer()

    def forward(self, hit_table_hit_id, hit_table_local_plane, hit_table_local_time, \
                    hit_table_local_wire, hit_table_integral, hit_table_rms, \
                    spacepoint_table_spacepoint_id, spacepoint_table_hit_id_u, spacepoint_table_hit_id_v, \
                    spacepoint_table_hit_id_y):
        
        gnn_hetero_data = self.hitgraph.create_graph(hit_table_hit_id, hit_table_local_plane, hit_table_local_time, \
                                                    hit_table_local_wire, hit_table_integral, hit_table_rms, \
                                                    spacepoint_table_spacepoint_id, spacepoint_table_hit_id_u, spacepoint_table_hit_id_v, \
                                                    spacepoint_table_hit_id_y)

        transform = Compose((ng.util.PositionFeatures(self.planes),
                             ng.util.FeatureNorm(self.planes, self.norm),
                             ng.util.FeatureExtension.FeatureExtension(self.planes)
                           ))
        hetero_dataset = HeteroDataset(gnn_hetero_data, transform=transform)
        data = hetero_dataset.get()
        self.model.step(data.to(self.device))
        x = self.model.data
        return  x['u']['x_semantic'].cpu().detach().numpy(), \
                x['v']['x_semantic'].cpu().detach().numpy(), x['y']['x_semantic'].cpu().detach().numpy(), \
                x['u']['x_filter'].cpu().detach().numpy(), x['v']['x_filter'].cpu().detach().numpy(), \
                x['y']['x_filter'].cpu().detach().numpy()

class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """

    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to initialize any state associated with this model.

        Parameters
        ----------
        args : dict
          Both keys and values are strings. The dictionary keys and values are:
          * model_config: A JSON string containing the model configuration
          * model_instance_kind: A string containing model instance kind
          * model_instance_device_id: A string containing model instance device ID
          * model_repository: Model repository path
          * model_version: Model version
          * model_name: Model name
        """

        # You must parse model_config. JSON string is not parsed here
        self.model_config = model_config = json.loads(args["model_config"])

        # Get ouptput configuration
        x_semantic_u_config = pb_utils.get_output_config_by_name(model_config, "x_semantic_u")
        x_semantic_v_config = pb_utils.get_output_config_by_name(model_config, "x_semantic_v")
        x_semantic_y_config = pb_utils.get_output_config_by_name(model_config, "x_semantic_y")
        x_filter_u_config = pb_utils.get_output_config_by_name(model_config, "x_filter_u")
        x_filter_v_config = pb_utils.get_output_config_by_name(model_config, "x_filter_v")
        x_filter_y_config = pb_utils.get_output_config_by_name(model_config, "x_filter_y")

        # Convert Triton types to numpy types
        self.x_semantic_u_dtype = pb_utils.triton_string_to_numpy(
            x_semantic_u_config["data_type"]
        )
        self.x_semantic_v_dtype = pb_utils.triton_string_to_numpy(
            x_semantic_v_config["data_type"]
        )
        self.x_semantic_y_dtype = pb_utils.triton_string_to_numpy(
            x_semantic_y_config["data_type"]
        )
        self.x_filter_u_dtype = pb_utils.triton_string_to_numpy(
            x_filter_u_config["data_type"]
        )
        self.x_filter_v_dtype = pb_utils.triton_string_to_numpy(
            x_filter_v_config["data_type"]
        )
        self.x_filter_y_dtype = pb_utils.triton_string_to_numpy(
            x_filter_y_config["data_type"]
        )
        # Instantiate the PyTorch model
        self.NuGraph2_model = NuGraph2_model()

    def execute(self, requests):
        """`execute` must be implemented in every Python model. `execute`
        function receives a list of pb_utils.InferenceRequest as the only
        argument. This function is called when an inference is requested
        for this model. Depending on the batching configuration (e.g. Dynamic
        Batching) used, `requests` may contain multiple requests. Every
        Python model, must create one pb_utils.InferenceResponse for every
        pb_utils.InferenceRequest in `requests`. If there is an error, you can
        set the error argument when creating a pb_utils.InferenceResponse.

        Parameters
        ----------
        requests : list
          A list of pb_utils.InferenceRequest

        Returns
        -------
        list
          A list of pb_utils.InferenceResponse. The length of this list must
          be the same as `requests`
        """

        x_semantic_u_dtype = self.x_semantic_u_dtype
        x_semantic_v_dtype = self.x_semantic_v_dtype
        x_semantic_y_dtype = self.x_semantic_y_dtype
        x_filter_u_dtype = self.x_filter_u_dtype
        x_filter_v_dtype = self.x_filter_v_dtype
        x_filter_y_dtype = self.x_filter_y_dtype


        responses = []

        # Every Python backend must iterate over everyone of the requests
        # and create a pb_utils.InferenceResponse for each of them.
        for request in requests:
            # Get all inputs
            hit_table_hit_id = pb_utils.get_input_tensor_by_name(request, "hit_table_hit_id")
            hit_table_local_plane = pb_utils.get_input_tensor_by_name(request, "hit_table_local_plane")
            hit_table_local_time = pb_utils.get_input_tensor_by_name(request, "hit_table_local_time")
            hit_table_local_wire = pb_utils.get_input_tensor_by_name(request, "hit_table_local_wire")
            hit_table_integral = pb_utils.get_input_tensor_by_name(request, "hit_table_integral")
            hit_table_rms = pb_utils.get_input_tensor_by_name(request, "hit_table_rms")

            spacepoint_table_spacepoint_id = pb_utils.get_input_tensor_by_name(request, "spacepoint_table_spacepoint_id")
            spacepoint_table_hit_id_u = pb_utils.get_input_tensor_by_name(request, "spacepoint_table_hit_id_u")
            spacepoint_table_hit_id_v = pb_utils.get_input_tensor_by_name(request, "spacepoint_table_hit_id_v")
            spacepoint_table_hit_id_y = pb_utils.get_input_tensor_by_name(request, "spacepoint_table_hit_id_y")
            
            output1, output2, output3, output4, output5, output6 = \
                                        self.NuGraph2_model(hit_table_hit_id.as_numpy(), hit_table_local_plane.as_numpy(), \
                                                    hit_table_local_time.as_numpy(), \
                    hit_table_local_wire.as_numpy(), hit_table_integral.as_numpy(), hit_table_rms.as_numpy(), \
                    spacepoint_table_spacepoint_id.as_numpy(), spacepoint_table_hit_id_u.as_numpy(), spacepoint_table_hit_id_v.as_numpy(), \
                    spacepoint_table_hit_id_y.as_numpy())
        

            # Create output tensors. You need pb_utils.Tensor
            # objects to create pb_utils.InferenceResponse.
            out_tensor_1 = pb_utils.Tensor("x_semantic_u", output1.astype(x_semantic_u_dtype))
            out_tensor_2 = pb_utils.Tensor("x_semantic_v", output2.astype(x_semantic_v_dtype))
            out_tensor_3 = pb_utils.Tensor("x_semantic_y", output3.astype(x_semantic_y_dtype))
            out_tensor_4 = pb_utils.Tensor("x_filter_u", output4.astype(x_filter_u_dtype))
            out_tensor_5 = pb_utils.Tensor("x_filter_v", output5.astype(x_filter_v_dtype))
            out_tensor_6 = pb_utils.Tensor("x_filter_y", output6.astype(x_filter_y_dtype))

            inference_response = pb_utils.InferenceResponse(
                output_tensors=[out_tensor_1, out_tensor_2, out_tensor_3, out_tensor_4, out_tensor_5, out_tensor_6]
            )
            responses.append(inference_response)

        # You should return a list of pb_utils.InferenceResponse. Length
        # of this list must match the length of `requests` list.
        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is optional. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print("Cleaning up...")
