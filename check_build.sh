# PyType must be run separately for each top-level directory.
# This is because there are files with identical names in them
# and that's a general issue for more involved (type) linters.
# pytype Machine-Learning/
# pytype Neural-Networks/
# pytype Deep-Learning/

# Style linters look over cosmetics in code and docs
flake8 Machine-Learning/ Neural-Networks/ Deep-Learning/
pydocstyle Machine-Learning/ Neural-Networks/ Deep-Learning/
