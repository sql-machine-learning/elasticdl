package common

import (
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	"log"
)

// GetMasterPodName returns master pod name
func GetMasterPodName(jobName string) string {
	return "elasticdl-" + jobName + "-master"
}

// CreateClientSet uses in-cluster config to create a clientset.
func CreateClientSet() *kubernetes.Clientset {
	config, err := rest.InClusterConfig()
	if err != nil {
		return nil
	}
	clientSet, err := kubernetes.NewForConfig(config)
	if err != nil {
		return nil
	}
	return clientSet
}

// PodFinished returns true if the pod is Succeeded/Failed, or still running but with metadata.labels["status"]==""Finished"
func PodFinished(clientSet *kubernetes.Clientset, namespace string, podName string) bool {
	pod, err := clientSet.CoreV1().Pods(namespace).Get(podName, metav1.GetOptions{})
	// Network instability may cause error, assuming not finished.
	if err != nil {
		log.Println("Get pod ", podName, " failed with err ", err)
		return false
	}
	if pod.Status.Phase == v1.PodFailed || pod.Status.Phase == v1.PodSucceeded {
		return true
	} else if pod.Status.Phase == v1.PodRunning {
		finished, ok := pod.ObjectMeta.Labels["status"]
		if ok && finished == "Finished" {
			return true
		}
	}
	return false
}
