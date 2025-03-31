package embedding

import "math/rand"

type Embeddings struct {
	dimension   *int
	rows        [][]float64
	seed        *int64
	growBy      *int
	rnd         *rand.Rand
	maxCapacity *int
}

func (o *Embeddings) grow() {
	rows := make([][]float64, *o.growBy)
	for i := 0; i < *o.growBy; i++ {
		row := make([]float64, *o.dimension)
		for j := 0; j < *o.dimension; j++ {
			n := o.rnd.Intn(*o.maxCapacity)
			row[j] = 1.0 / float64(n)
		}
		rows[i] = row
	}
	o.rows = append(o.rows, rows...)
}

func (o *Embeddings) At(index int) []float64 {
	for index >= len(o.rows) {
		o.grow()
	}
	return o.rows[index]
}

func (o *Embeddings) SequenceLookup(indices []int, length int) [][]float64 {
	result := make([][]float64, length)
	l := len(indices)
	for i := 0; i < length; i++ {
		if i < l {
			result[i] = o.At(indices[i])
		} else {
			result[i] = make([]float64, *o.dimension)
		}
	}
	return result
}

func (o *Embeddings) BatchLookup(batches [][]int, length int) [][][]float64 {
	result := make([][][]float64, len(batches))
	for i, b := range batches {
		seq := o.SequenceLookup(b, length)
		result[i] = seq
	}
	return result
}

func NewEmbeddings(dimension int, maxCapacity int, growBy int, seed int64) *Embeddings {
	source := rand.NewSource(seed)
	r := rand.New(source)
	return &Embeddings{dimension: &dimension, growBy: &growBy,
		rnd: r, maxCapacity: &maxCapacity}
}
