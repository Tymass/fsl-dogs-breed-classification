#!/bin/bash

echo $1
tensorboard --logdir $1 & xdg-open http://localhost:6006/

