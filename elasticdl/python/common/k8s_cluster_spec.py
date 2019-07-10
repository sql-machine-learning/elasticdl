class KubernetesCluster:
    def with_cluster(self, pod):
        # By default, don't need to add specific config for pod
        return pod


cluster = KubernetesCluster()
