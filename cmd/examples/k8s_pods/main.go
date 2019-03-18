package main

import (
	"flag"
	"fmt"
	"time"

	apiv1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	watch "k8s.io/apimachinery/pkg/watch"
	"k8s.io/client-go/kubernetes"
	v1 "k8s.io/client-go/kubernetes/typed/core/v1"
	"k8s.io/client-go/tools/clientcmd"
)

func createPodClient(kubeconfig string) v1.PodInterface {
	config, err := clientcmd.BuildConfigFromFlags("", kubeconfig)
	if err != nil {
		panic(err.Error())
	}
	clientset, err := kubernetes.NewForConfig(config)
	if err != nil {
		panic(err.Error())
	}

	return clientset.CoreV1().Pods(apiv1.NamespaceDefault)
}

func watchPodEvents(podclient v1.PodInterface) {
	watcher, err := podclient.Watch(metav1.ListOptions{})
	if err != nil {
		panic(err.Error())
	}

	// Watch for POD events.
	go func(ch <-chan watch.Event) {
		for event := range ch {
			fmt.Println(event)
		}
	}(watcher.ResultChan())
}

func deletePodIfExists(podclient v1.PodInterface, name string) {
	// List all PODs
	pods, err := podclient.List(metav1.ListOptions{})
	if err != nil {
		panic(err.Error())
	}
	for _, p := range pods.Items {
		fmt.Println(p.Name)
		// Kill POD if exists
		if p.Name == name {
			err := podclient.Delete(p.Name, &metav1.DeleteOptions{})
			if err != nil {
				panic(err.Error())
			}
			// Wait for a while for k8s to finalize.
			time.Sleep(30 * time.Second)
		}
	}
}

func createPod(podclient v1.PodInterface, name string) {
	pod := &apiv1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
		Spec: apiv1.PodSpec{
			Containers: []apiv1.Container{
				{
					Name:  "poc-container1",
					Image: "nginx:1.12",
					Ports: []apiv1.ContainerPort{
						{
							Name:          "http",
							Protocol:      apiv1.ProtocolTCP,
							ContainerPort: 80,
						},
					},
				},
			},
		},
	}
	_, err := podclient.Create(pod)
	if err != nil {
		panic(err.Error())
	}
}

func printPodNames(podclient v1.PodInterface) {
	pods, err := podclient.List(metav1.ListOptions{})
	if err != nil {
		panic(err.Error())
	}
	for _, p := range pods.Items {
		fmt.Println(p.Name)
	}
}

func main() {
	var kubeconfig = flag.String("kubeconfig", "", "Path to kubeconfig file")
	flag.Parse()

	podclient := createPodClient(*kubeconfig)
	watchPodEvents(podclient)

	podname := "pod-example"
	deletePodIfExists(podclient, podname)
	createPod(podclient, podname)

	printPodNames(podclient)
}
