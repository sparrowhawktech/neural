package embedding

import "math/rand"

func Dropout(batch [][]float64, dimension int, p float64, seed int64) {
	total := len(batch) * dimension
	max := int(float64(total) * p)
	rnd := rand.New(rand.NewSource(seed))
	locations := make([]int, max)
	for i := 0; i < max; i++ {
		location := rnd.Intn(total)
		l := locations[i]
		for l == location {
			location = rnd.Intn(total)
		}
		locations[i] = location
	}

	for _, l := range locations {
		x := l / dimension
		y := l - (x * dimension)
		batch[x][y] = 0
	}

}
