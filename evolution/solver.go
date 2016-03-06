package evolution

import (
	"math/rand"
	"sort"
)

type Solver struct {
	StepCount       int
	StepSizeInitial float64
	StepSizeFinal   float64

	MaxPopulation        int
	MutateProbability    float64
	CrossOverProbability float64
	SelectionProbability float64

	DFTradeoff DFTradeoff
}

// Solve runs evolution using the given parameters and then returns a list of entities, sorted from
// most fit to least fit.
func (s *Solver) Solve(start []Entity) []Entity {
	population := start
	for i := 0; i < s.StepCount; i++ {
		var fracDone float64
		if s.StepCount == 1 {
			fracDone = 1
		} else {
			fracDone = float64(i) / float64(s.StepCount-1)
		}
		newPool := s.nextOffspring(population, fracDone)
		population = s.applySelection(newPool)
	}
	sortEntities(population, nil, nil)
	return population
}

func (s *Solver) nextOffspring(population []Entity, fracDone float64) []Entity {
	stepSize := s.StepSizeFinal*fracDone + s.StepSizeInitial*(1-fracDone)
	newPool := make([]Entity, len(population))
	copy(newPool, population)
	for j, entity := range population {
		if rand.Float64() < s.MutateProbability {
			newPool = append(newPool, entity.Mutate(stepSize))
		}
		if rand.Float64() < s.CrossOverProbability && len(population) > 1 {
			mateIdx := rand.Intn(len(population))
			for mateIdx == j {
				mateIdx = rand.Intn(len(population))
			}
			mate := population[mateIdx]
			newPool = append(newPool, entity.CrossOver(mate))
		}
	}
	return newPool
}

func (s *Solver) applySelection(population []Entity) []Entity {
	selected := make([]Entity, 0, s.MaxPopulation)
	remaining := make([]Entity, len(population))
	copy(remaining, population)

	for i := 0; i < s.MaxPopulation && len(remaining) > 0; i++ {
		sortEntities(remaining, selected, s.DFTradeoff)
		selectedIdx := len(remaining) - 1
		selectedEntity := remaining[selectedIdx]
		for j, ent := range remaining[:len(remaining)-1] {
			if rand.Float64() < s.SelectionProbability {
				selectedEntity = ent
				selectedIdx = j
				break
			}
		}
		selected = append(selected, selectedEntity)
		remaining[selectedIdx] = remaining[len(remaining)-1]
		remaining = remaining[:len(remaining)-1]
	}

	return selected
}

func sortEntities(entities, selected []Entity, tradeoff DFTradeoff) {
	if len(entities) < 2 {
		return
	}

	sorter := &entitySorter{
		selected:      nil,
		remaining:     entities,
		fitnessRank:   make([]int, len(entities)),
		diversityRank: make([]int, len(entities)),
		tradeoff:      tradeoff,
	}

	sort.Sort(sorter)

	if len(selected) == 0 {
		return
	}

	for i := range entities {
		sorter.fitnessRank[i] = i
	}

	if len(selected) > 0 {
		sorter.selected = selected
		sorter.findDiversityRank = true
		sorter.similarityCache = make([]float64, len(entities))
		for i, ent := range sorter.remaining {
			sorter.similarityCache[i] = ent.Similarity(selected)
		}
		sort.Sort(sorter)
		for i := range entities {
			sorter.diversityRank[i] = i
		}
		sorter.findDiversityRank = false
		sort.Sort(sorter)
	}
}

type entitySorter struct {
	selected      []Entity
	remaining     []Entity
	fitnessRank   []int
	diversityRank []int
	tradeoff      DFTradeoff

	findDiversityRank bool
	similarityCache   []float64
}

func (e *entitySorter) Len() int {
	return len(e.remaining)
}

func (e *entitySorter) Swap(i, j int) {
	e.remaining[i], e.remaining[j] = e.remaining[j], e.remaining[i]
	e.fitnessRank[i], e.fitnessRank[j] = e.fitnessRank[j], e.fitnessRank[i]
	e.diversityRank[i], e.diversityRank[j] = e.diversityRank[j], e.diversityRank[i]
	if e.similarityCache != nil {
		e.similarityCache[i], e.similarityCache[j] = e.similarityCache[j], e.similarityCache[i]
	}
}

func (e *entitySorter) Less(i, j int) bool {
	e1 := e.remaining[i]
	e2 := e.remaining[j]
	if len(e.selected) == 0 {
		return e1.Fitness() > e2.Fitness()
	} else if e.findDiversityRank {
		return e.similarityCache[i] < e.similarityCache[j]
	}
	f1 := e.floatingFitnessRank(i)
	d1 := e.floatingDiversityRank(i)
	f2 := e.floatingFitnessRank(j)
	d2 := e.floatingDiversityRank(j)

	goodness1 := e.tradeoff(d1, f1)
	goodness2 := e.tradeoff(d2, f2)
	return goodness1 > goodness2
}

func (e *entitySorter) floatingFitnessRank(i int) float64 {
	rank := e.fitnessRank[i]
	reverseRank := len(e.fitnessRank) - (rank + 1)
	return float64(reverseRank) / float64(len(e.fitnessRank)-1)
}

func (e *entitySorter) floatingDiversityRank(i int) float64 {
	rank := e.diversityRank[i]
	reverseRank := len(e.diversityRank) - (rank + 1)
	return float64(reverseRank) / float64(len(e.diversityRank)-1)
}
