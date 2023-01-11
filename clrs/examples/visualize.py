import clrs
from clrs.examples.run import create_samplers
from absl import flags
import jax
import numpy as np

FLAGS = flags.FLAGS

def load_model(weight_file):
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
      )

    eval_model = clrs.models.BaselineModel(
        spec=spec_list,
        dummy_trajectory=[next(val_samplers[0][0])],
        **model_params
    ) 

    eval_model.restore_model(weight_file)
    return eval_model
    

if __name__ == "__main__":
    model = load_model("../../clrs_results/bellman_ford/bellman_ford_epoch_8192.pkl")
    print(model)
    print(model.layers)