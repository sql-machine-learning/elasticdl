if [ $# -gt 0 ]; then
    train_data_dir=$1
    /home/steven.cheng/install/python3.5/bin/python3 ../launcher.py "mnist.py" \
                 "--class_name"  "MnistCNN" \
                 "--runner"      "thread" \
                 "--num_ps"      "1" \
                 "--num_worker"  "1" \
                 "--input"       ${train_data_dir} 
else
    echo "train data dir needed!"
    exit 1
fi
