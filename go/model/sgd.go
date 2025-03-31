package model

import "neural/backend"

type SGD struct {
	Alpha           *float64
	CostActivatorId *int
	network         *Network
	api             backend.TensorsApi
}

func (o *SGD) Backward() {
	o.api.CostDerivative(*o.CostActivatorId, *o.network.OutputLayer.OutputValuesHandle, *o.network.TruthsHandle, *o.network.OutputLayer.DeltasHandle)
	o.computeDeltas(o.network.OutputLayer, o.network.OutputLayer.SourceLayer)
	o.updateWeights(o.network.OutputLayer)
}

func (o *SGD) computeDeltas(l1, l0 *DenseLayer) {
	o.api.ComputeDeltas(*l1.DeltasHandle, *l1.WeightsHandle, *l0.ValuesHandle, *l0.activatorId, *l0.DeltasHandle)
	if l0.SourceLayer != nil {
		o.computeDeltas(l0, l0.SourceLayer)
	}
}

func (o *SGD) Forward() {
	all := make([]*DenseLayer, 0)
	l := o.network.OutputLayer
	for {
		all = append(all, l)
		l = l.SourceLayer
		if l == nil {
			break
		}
	}
	for i := len(all) - 2; i > -1; i-- {
		o.forwardLayer(all[i])
	}
	for i := 0; i < len(o.network.Costs); i++ {
		o.network.Costs[i] = 0
	}
	o.network.Cost = o.api.Cost(*o.CostActivatorId, *o.network.OutputLayer.OutputValuesHandle, *o.network.TruthsHandle, o.network.Costs)
}

func (o *SGD) updateWeights(layer *DenseLayer) {
	o.api.UpdateWeights(*layer.SourceLayer.OutputValuesHandle, *layer.WeightsHandle, *layer.DeltasHandle, *o.Alpha)
	if layer.SourceLayer.WeightsHandle != nil {
		o.updateWeights(layer.SourceLayer)
	}
}

func (o *SGD) forwardLayer(layer *DenseLayer) {
	o.api.Forward(*layer.SourceLayer.OutputValuesHandle, *layer.WeightsHandle, *layer.ValuesHandle,
		*layer.DeltasHandle,
		*layer.OutputValuesHandle, *layer.activatorId)
}

func NewSGD(network *Network, api backend.TensorsApi, costActivatorId int, alpha float64) *SGD {
	return &SGD{
		network:         network,
		CostActivatorId: &costActivatorId,
		Alpha:           &alpha,
		api:             api,
	}
}
