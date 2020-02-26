# Do all runs:

# 'baseline'
# 'linear', 'sigmoid', 'softplus', 'elu'
# 'sgd', 'rmsprop', 'adadelta'
# 'short_cnn', 'long_cnn', 'small_model'
# 'none', 'high'

# Baseline
python CNN.py baseline

# Optimizers in {'sgd', 'rmsprop', 'adadelta'}
python CNN.py --optimizer sgd       opt_sgd
python CNN.py --optimizer rmsprop   opt_rmsprop
python CNN.py --optimizer adadelta  opt_adadelta

# Architectures in {'short_cnn', 'long_cnn', 'small_model'}
python CNN.py --architecture short_cnn  arch_short
python CNN.py --architecture long_cnn   arch_long
python CNN.py --architecture small_cnn  arch_small

# Activations in {'linear', 'sigmoid', 'softplus', 'elu'}
python CNN.py --activation linear    act_linear
python CNN.py --activation sigmoid   act_sigmoid
python CNN.py --activation softplus  act_softplus
python CNN.py --activation elu       act_elu

# Dropouts in {'none', 'high'}
python CNN.py --dropout none  dropout_none
python CNN.py --dropout high  dropout_high

