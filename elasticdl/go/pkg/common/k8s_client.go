// Copyright 2020 The SQLFlow Authors. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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
