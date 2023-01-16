import clrs
from clrs.examples.run import create_samplers
from absl import flags
import jax
import numpy as np
from absl import app
import pickle 
import jax.numpy as jnp
import haiku as hk
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import re

FLAGS = flags.FLAGS

def load_model(weight_file):
    epoch = re.findall("_([0-9]+).pkl", weight_file)[-1]
    ACTIVATIONS = {"edge_fts": [],
                "node_fts": [],
                "graph_fts": [],
                "nxt_hidden": [],
                "nxt_edge": []}

    def ec(edge_fts, node_fts, graph_fts):
        # print("Encode callback called")
        # print(f"edge_fts: {edge_fts.shape}")
        # print(f"node_fts: {node_fts.shape}")
        # print(f"graph_fts: {graph_fts.shape}")
        ACTIVATIONS['edge_fts'].append(edge_fts)
        ACTIVATIONS['node_fts'].append(node_fts)
        ACTIVATIONS['graph_fts'].append(graph_fts)

    def pc(nxt_hidden, nxt_edge):
        # print("Process callback called")
        # print(f"nxt_hidden: {nxt_hidden.shape}")
        # print(f"nxt_edge: {nxt_edge.shape}")
        ACTIVATIONS["nxt_edge"].append(nxt_edge)
        ACTIVATIONS["nxt_hidden"].append(nxt_hidden)


    train_lengths = [int(x) for x in FLAGS.train_lengths]
    val_lengths = [int(x) for x in FLAGS.val_lengths]
    rng = np.random.RandomState(FLAGS.seed)
    rng_key = jax.random.PRNGKey(rng.randint(2**32))
    (train_samplers,
    val_samplers, val_sample_counts,
    test_samplers, test_sample_counts,
    spec_list) = create_samplers(rng, train_lengths, val_lengths)

    processor_factory = clrs.get_processor_factory(
        FLAGS.processor_type,
        use_ln=FLAGS.use_ln,
        nb_triplet_fts=FLAGS.nb_triplet_fts,
        nb_heads=FLAGS.nb_heads
    )   

    if FLAGS.hint_mode == 'encoded_decoded':
        encode_hints = True
        decode_hints = True
    elif FLAGS.hint_mode == 'decoded_only':
        encode_hints = False
        decode_hints = True
    elif FLAGS.hint_mode == 'none':
        encode_hints = False
        decode_hints = False
    else:
        raise ValueError('Hint mode not in {encoded_decoded, decoded_only, none}.')


    model_params = dict(
      processor_factory=processor_factory,
      hidden_dim=FLAGS.hidden_size,
      encode_hints=encode_hints,
      decode_hints=decode_hints,
      encoder_init=FLAGS.encoder_init,
      use_lstm=FLAGS.use_lstm,
      learning_rate=FLAGS.learning_rate,
      grad_clip_max_norm=FLAGS.grad_clip_max_norm,
      checkpoint_path=FLAGS.checkpoint_path,
      freeze_processor=FLAGS.freeze_processor,
      dropout_prob=FLAGS.dropout_prob,
      hint_teacher_forcing=FLAGS.hint_teacher_forcing,
      hint_repred_mode=FLAGS.hint_repred_mode,
      nb_msg_passing_steps=FLAGS.nb_msg_passing_steps,
      encode_callbacks=[ec],
      process_callbacks=[pc]
      )

    model = clrs.models.BaselineModel(
        spec=spec_list,
        dummy_trajectory=[next(val_samplers[0][0])],
        **model_params
    ) 

    feedback_list = [next(t) for t in train_samplers]
    all_features = [f.features for f in feedback_list]
    model.init(all_features, FLAGS.seed + 1, FLAGS.alpha)

    # eval_model._restore_model(weight_file)
    with open(weight_file, 'rb') as f:
        restored_state = pickle.load(f)
        restored_params = restored_state['params']
        # print(model.params)
        model.params = hk.data_structures.merge(model.params, restored_params)
        model.opt_state = restored_state['opt_state']
        # return model

    count = train_lengths[0]
    sampler = train_samplers[0]
    processed_samples = 0
    while processed_samples < count:
        feedback = next(sampler)
        batch_size = feedback.outputs[0].data.shape[0]
        new_rng_key, rng_key = jax.random.split(rng_key)
        model.predict(new_rng_key, feedback.features, 0, return_hints=True)
        processed_samples += batch_size
    for key in ACTIVATIONS.keys():
        arr = jnp.concatenate(ACTIVATIONS[key], axis=-1)
        ACTIVATIONS[key] = np.array(arr.reshape(-1, arr.shape[-1]))
        print(key, ACTIVATIONS[key].shape)
        name = key + "_epoch_" + epoch
        pca = PCA(n_components=2)
        projected = pca.fit_transform(ACTIVATIONS[key])
        plot_pca(name, projected)
        pca2 = PCA(n_components=50)
        projected2 = pca2.fit_transform(ACTIVATIONS[key])
        plot_tsne(projected2, [i for i in range(projected.shape[0])],name)

def plot_pca(name, projected):
    f = plt.figure(figsize=(13, 13))
    ax = plt.subplot(aspect='equal') 
    ax.scatter(projected[:, 0], projected[:, 1], edgecolor='none', alpha=0.5)
    ax.set_xlabel('component 1')
    ax.set_ylabel('component 2')
    ax.axis('tight')
    save_path = "clrs/figures/" + name + "_pca_plot"
    plt.savefig(save_path)
    print("Saved plot to " + save_path )

def plot_tsne(x, colours , name):
    cm = plt.cm.get_cmap('RdYlGn')
    f = plt.figure(figsize=(13, 13))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40, c=colours, cmap=cm)
    # if labels is not None:
    #     for i, txt in enumerate(labels):
    #         ax.annotate(txt, (x[i,0], x[i,1]))
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    # ax.axis('off')
    ax.axis('tight')
    save_path = "clrs/figures/" + name + "_tsne_plot"
    plt.savefig(save_path)


def main(unused_argv):
    load_model("results/bellman_ford/bellman_ford_epoch_8192.pkl")
    load_model("results/bellman_ford/bellman_ford_epoch_1.pkl")

if __name__ == "__main__":
    app.run(main)
    