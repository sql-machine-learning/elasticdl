package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"time"

	apiv1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	watch "k8s.io/apimachinery/pkg/watch"
	"k8s.io/client-go/kubernetes"
	v1 "k8s.io/client-go/kubernetes/typed/core/v1"
	"k8s.io/client-go/tools/clientcmd"
	metricsv1beta1 "k8s.io/metrics/pkg/client/clientset/versioned/typed/metrics/v1beta1"
)

func formatJson(in interface{}) string {
	out, err := json.MarshalIndent(in, "", "  ")
	if err != nil {
		panic(err.Error())
	}
	return string(out)
}

func createPodClients(kubeconfig string) (v1.PodInterface, metricsv1beta1.PodMetricsInterface) {
	config, err := clientcmd.BuildConfigFromFlags("", kubeconfig)
	if err != nil {
		panic(err.Error())
	}
	clientset, err := kubernetes.NewForConfig(config)
	if err != nil {
		panic(err.Error())
	}

	metricsclient, err := metricsv1beta1.NewForConfig(config)
	if err != nil {
		panic(err.Error())
	}

	return clientset.CoreV1().Pods(apiv1.NamespaceDefault), metricsclient.PodMetricses(apiv1.NamespaceDefault)
}

func watchPodEvents(podclient v1.PodInterface) {
	watcher, err := podclient.Watch(metav1.ListOptions{})
	if err != nil {
		panic(err.Error())
	}

	// Watch for POD events.
	go func(ch <-chan watch.Event) {
		for event := range ch {
			fmt.Println(formatJson(event))
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
	// Wait for a while for POD to start.
	time.Sleep(30 * time.Second)
}

func printPodMetrics(podclient v1.PodInterface, metricsclient metricsv1beta1.PodMetricsInterface) {
	pods, err := podclient.List(metav1.ListOptions{})
	if err != nil {
		panic(err.Error())
	}
	for _, p := range pods.Items {
		metrics, err := metricsclient.Get(p.Name, metav1.GetOptions{})
		if err != nil {
			fmt.Printf("%s %s\n", p.Name, err.Error())
		} else {
			fmt.Println(formatJson(*metrics))
		}
	}
}

func main() {
	var kubeconfig = flag.String("kubeconfig", "", "Path to kubeconfig file")
	flag.Parse()

	podclient, metricsclient := createPodClients(*kubeconfig)
	watchPodEvents(podclient)

	podname := "pod-example"
	deletePodIfExists(podclient, podname)
	createPod(podclient, podname)

	fmt.Println("----------")
	printPodMetrics(podclient, metricsclient)
}
