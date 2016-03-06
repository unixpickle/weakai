package evolution

// A DFTradeoff returns a "goodness" value, the higher the better, for an entity given its diversity
// rank and fitness rank.
// Both diversity rank and fitness rank are between 0 and 1, where 1 indicates the most fitness or
// diversity.
type DFTradeoff func(diversity, fitness float64) float64

// LinearDFTradeoff returns the sum of the diversity rank and the fitness rank values.
func LinearDFTradeoff(diversity, fitness float64) float64 {
	return diversity + fitness
}
