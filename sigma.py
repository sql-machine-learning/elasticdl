from kubernetes import client


class Sigma:
    def __init__(self):
        self._cluster = "eu95"
        self._pool = "rdma"

    def with_pod(self, pod):
        self._with_labels(pod)
        self._with_tolerations(pod)
        self._with_affinity(pod)
        return pod

    def with_service(self, service):
        service.metadata.annotations["service.k8s.alipay.com/provisioner"] = "xvip"
        service.metadata.annotations["service.k8s.alipay.com/xvip-app-group"] = "sigma-alipay-test"
        service.metadata.annotations["service.k8s.alipay.com/xvip-bu-type"] = "internal"
        service.metadata.annotations["service.k8s.alipay.com/xvip-healthcheck-type"] = "TCPCHECK"
        service.metadata.annotations["service.k8s.alipay.com/xvip-qps-limit"] = "1024"
        service.metadata.annotations["service.k8s.alipay.com/xvip-req-avg-size"] = "1024"
        return service

    def _with_labels(self, pod):
        pod.metadata.labels["sigma.ali/app-name"] = "elasticdl"
        pod.metadata.labels["sigma.ali/deploy-unit"] = "elasticdl"
        pod.metadata.labels["sigma.ali/instance-group"] = "sigma-alipay-test"
        pod.metadata.labels["sigma.ali/site"] = self._cluster

    def _with_tolerations(self, pod):
        tolerations = [
            client.V1Toleration(
                effect="NoSchedule",
                key="mandatory.k8s.alipay.com/app-logic-pool",
                operator="Equal",
                value="kubemaker",
            )
        ]
        if self._pool is not "kubemaker":
            tolerations.append(
                client.V1Toleration(
                    effect="NoSchedule",
                    key="mandatory.k8s.alipay.com/server-owner",
                    operator="Equal",
                    value=self._pool,
                )
            )
        pod.spec.tolerations = tolerations

    def _with_affinity(self, pod):
        match_expressions = [
            client.V1NodeSelectorRequirement(
                key="mandatory.k8s.alipay.com/app-logic-pool",
                operator="In",
                values=["kubemaker"],
            ),
        ]
        if self._pool is not "kubemaker":
            match_expressions.append(
                client.V1NodeSelectorRequirement(
                    key="mandatory.k8s.alipay.com/server-owner",
                    operator="In",
                    values=[self._pool],
                )
            )
        affinity = client.V1Affinity(
            node_affinity=client.V1NodeAffinity(
                required_during_scheduling_ignored_during_execution=client.V1NodeSelector(
                    node_selector_terms=[
                        client.V1NodeSelectorTerm(
                            match_expressions=match_expressions,
                        )
                    ]
                )
            )
        )
        pod.spec.affinity = affinity


cluster = Sigma()
