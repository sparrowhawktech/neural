package model

import (
	"math/rand"
	"neural/backend"
	"sparrowhawktech/toolkit/util"
)

type DenseLayer struct {
	Name               *string
	rnd                *rand.Rand
	Dimension          *int
	WeightsHandle      *int
	ValuesHandle       *int
	OutputValuesHandle *int
	DeltasHandle       *int
	Cardinality        *int
	SourceLayer        *DenseLayer
	api                backend.TensorsApi
	activatorId        *int
	seed               *int64
}

func (o *DenseLayer) Connect(sourceLayer *DenseLayer) {
	inputDimension := *sourceLayer.Dimension
	o.WeightsHandle = util.PInt(o.api.BuildWeights(*o.Dimension, inputDimension, *o.seed))
	o.ValuesHandle = util.PInt(o.api.CreateTensor(*o.Dimension, *sourceLayer.Cardinality))
	o.OutputValuesHandle = util.PInt(o.api.CreateTensor(*o.Dimension, *sourceLayer.Cardinality))
	o.DeltasHandle = util.PInt(o.api.CreateTensor(*o.Dimension))
	o.Cardinality = sourceLayer.Cardinality
	o.SourceLayer = sourceLayer
}

func (o *DenseLayer) LoadValues(inputs [][]float64) {
	o.ValuesHandle = util.PInt(o.api.CreateTensor(*o.Dimension, len(inputs)))
	o.OutputValuesHandle = util.PInt(o.api.CreateTensor(*o.Dimension, len(inputs)))
	o.DeltasHandle = util.PInt(o.api.CreateTensor(*o.Dimension))
	o.api.LoadInputValues(*o.ValuesHandle, inputs)
	o.api.Copy(*o.ValuesHandle, *o.OutputValuesHandle)
	o.Cardinality = util.PInt(len(inputs))

}

func NewLayer(name string, api backend.TensorsApi, dimension int, seed int64, activatorId int) *DenseLayer {
	return &DenseLayer{
		Name:        &name,
		Dimension:   &dimension,
		seed:        &seed,
		api:         api,
		activatorId: &activatorId,
	}
}
