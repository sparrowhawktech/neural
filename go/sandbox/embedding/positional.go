package embedding

import "math"

type Encodings struct {
	Vectors   [][]float64
	dimension *int
	size      *int
	num       *int
}

func (o *Encodings) Init() {
	vectors := make([][]float64, *o.size)
	dm := float64(*o.dimension)
	n := float64(*o.num)
	for k := 0; k < *o.size; k++ {
		kf := float64(k)
		vector := make([]float64, *o.dimension)
		for i := 0; i < *o.dimension/2; i++ {
			di := float64(i)
			vector[2*i] = math.Sin(kf / (2 * di) / math.Pow(n, dm))
			vector[(2*i)+1] = math.Cos(kf / (2 * di) / math.Pow(n, dm))
		}
		vectors = append(vectors, vector)
	}
	o.Vectors = vectors
}

func NewEncodings(dimension int, size int, num int) *Encodings {
	return &Encodings{dimension: &dimension, size: &size, num: &num}
}

func BuildEncodings(dimension int, size int, num int) [][]float64 {
	vectors := make([][]float64, size)
	dm := float64(dimension)
	n := float64(num)
	for k := 0; k < size; k++ {
		kf := float64(k)
		vector := make([]float64, dimension)
		for i := 0; i < dimension/2; i++ {
			di := float64(i)
			theta := kf / math.Pow(n, (2*di)/dm)
			vector[2*i] = math.Sin(theta)
			vector[(2*i)+1] = math.Cos(theta)
		}
		vectors[k] = vector
	}
	return vectors
}

func AddEmbeddings(embeddings [][]float64, vectors [][]float64, dimension int) [][]float64 {
	l := len(embeddings)
	result := make([][]float64, l)
	for k := 0; k < l; k++ {
		pev := make([]float64, dimension)
		for i := 0; i < dimension; i++ {
			pev[i] = embeddings[k][i] + vectors[k][i]
		}
		result[k] = pev
	}
	return result
}
