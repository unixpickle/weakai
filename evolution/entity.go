package evolution

// An Entity is a potential solution that evolution can manipulate and cause to reproduce.
// Entities are immutable, so mutations and cross-over generate new Entities.
type Entity interface {
	// Fitness an arbitrary measure of how good a given Entity is.
	// The higher this value, the "better" the Entity.
	Fitness() float64

	// Similarity returns a measurement of how similar this entity is to a bunch of other entities.
	// The higher the value, the more similar this entity is.
	Similarity(e []Entity) float64

	// Mutate returns an entity which is a mutated form of the current one.
	// The stepSize argument is a number indicating how drastic the mutation should be.
	// The higher the stepSize, the bigger the mutation.
	Mutate(stepSize float64) Entity

	// CrossOver returns an entity which is a random-ish combination of this one and another one.
	CrossOver(e Entity) Entity
}
