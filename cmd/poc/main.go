package main

import (
	"flag"
	"fmt"

	apiv1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/tools/clientcmd"
)

func main() {
	var kubeconfig = flag.String("kubeconfig", "", "Path to kubeconfig file")
	flag.Parse()

	config, err := clientcmd.BuildConfigFromFlags("", *kubeconfig)
	if err != nil {
		panic(err.Error())
	}
	clientset, err := kubernetes.NewForConfig(config)
	if err != nil {
		panic(err.Error())
	}

	podclient := clientset.CoreV1().Pods("")
	pods, err := podclient.List(metav1.ListOptions{})
	if err != nil {
		panic(err.Error())
	}
	for _, p := range pods.Items {
		fmt.Println(p.Name)
	}

	pod := &apiv1.Pod{}
	p, err := podclient.Create(pod)
	if err != nil {
		panic(err.Error())
	}
	fmt.Println(p.Name)
}
