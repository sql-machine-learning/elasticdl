if [ $# -gt 0 ]; then
    train_data_dir=$1
    python3 ../launcher.py "mnist.py" \
                 "--class_name"  "MnistCNN" \
                 "--runner"      "thread" \
                 "--num_ps"      "2" \
                 "--num_worker"  "4" \
                 "--input"       ${train_data_dir} 
else
    echo "train data dir needed!"
    exit 1
fi
