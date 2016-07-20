package idtrees

import (
	"fmt"
	"testing"
)

func TestID3(t *testing.T) {
	tests := []*treeTest{
		&treeTest{
			Samples: []Sample{
				treeTestSample{"age": int64(3), "class": "child"},
				treeTestSample{"age": int64(4), "class": "child"},
				treeTestSample{"age": int64(5), "class": "child"},
				treeTestSample{"age": int64(6), "class": "child"},
				treeTestSample{"age": int64(28), "class": "adult"},
				treeTestSample{"age": int64(15), "class": "teenager"},
				treeTestSample{"age": int64(17), "class": "teenager"},
				treeTestSample{"age": int64(16), "class": "teenager"},
				treeTestSample{"age": int64(30), "class": "adult"},
			},
			Attrs: []string{"age"},
			Expected: &Tree{
				Attr: "age",
				NumSplit: &NumSplit{
					Threshold: int64(10),
					LessEqual: &Tree{
						Classification: map[interface{}]float64{"child": 1},
					},
					Greater: &Tree{
						Attr: "age",
						NumSplit: &NumSplit{
							Threshold: int64(22),
							LessEqual: &Tree{
								Classification: map[interface{}]float64{"teenager": 1},
							},
							Greater: &Tree{
								Classification: map[interface{}]float64{"adult": 1},
							},
						},
					},
				},
			},
		},
		&treeTest{
			Samples: []Sample{
				treeTestSample{"height": 2.0, "class": "child"},
				treeTestSample{"height": 3.0, "class": "child"},
				treeTestSample{"height": 2.3, "class": "child"},
				treeTestSample{"height": 2.9, "class": "child"},
				treeTestSample{"height": 5.5, "class": "adult"},
				treeTestSample{"height": 4.3, "class": "teenager"},
				treeTestSample{"height": 4.5, "class": "teenager"},
				treeTestSample{"height": 5.0, "class": "teenager"},
				treeTestSample{"height": 6.0, "class": "adult"},
			},
			Attrs: []string{"height"},
			Expected: &Tree{
				Attr: "height",
				NumSplit: &NumSplit{
					Threshold: (3.0 + 4.3) / 2.0,
					LessEqual: &Tree{
						Classification: map[interface{}]float64{"child": 1},
					},
					Greater: &Tree{
						Attr: "height",
						NumSplit: &NumSplit{
							Threshold: (5.0 + 5.5) / 2.0,
							LessEqual: &Tree{
								Classification: map[interface{}]float64{"teenager": 1},
							},
							Greater: &Tree{
								Classification: map[interface{}]float64{"adult": 1},
							},
						},
					},
				},
			},
		},
		&treeTest{
			Samples: []Sample{
				treeTestSample{"drinks": false, "height": 2.0, "class": "child"},
				treeTestSample{"drinks": false, "height": 3.0, "class": "child"},
				treeTestSample{"drinks": false, "height": 2.3, "class": "child"},
				treeTestSample{"drinks": false, "height": 2.9, "class": "child"},
				treeTestSample{"drinks": true, "height": 5.5, "class": "adult"},
				treeTestSample{"drinks": false, "height": 4.3, "class": "teenager"},
				treeTestSample{"drinks": false, "height": 5.5, "class": "teenager"},
				treeTestSample{"drinks": false, "height": 6.0, "class": "teenager"},
				treeTestSample{"drinks": true, "height": 6.0, "class": "adult"},
			},
			Attrs: []string{"height", "drinks"},
			Expected: &Tree{
				Attr: "height",
				NumSplit: &NumSplit{
					Threshold: (3.0 + 4.3) / 2.0,
					LessEqual: &Tree{
						Classification: map[interface{}]float64{"child": 1},
					},
					Greater: &Tree{
						Attr: "drinks",
						ValSplit: map[interface{}]*Tree{
							true: &Tree{
								Classification: map[interface{}]float64{"adult": 1},
							},
							false: &Tree{
								Classification: map[interface{}]float64{"teenager": 1},
							},
						},
					},
				},
			},
		},
		&treeTest{
			Samples: []Sample{
				treeTestSample{"drinks": false, "height": 2.0, "class": "child"},
				treeTestSample{"drinks": false, "height": 3.0, "class": "child"},
				treeTestSample{"drinks": false, "height": 2.3, "class": "child"},
				treeTestSample{"drinks": false, "height": 2.9, "class": "child"},
				treeTestSample{"drinks": true, "height": 5.5, "class": "adult"},
				treeTestSample{"drinks": true, "height": 5.3, "class": "adult"},
				treeTestSample{"drinks": false, "height": 4.3, "class": "teenager"},
				treeTestSample{"drinks": false, "height": 5.6, "class": "teenager"},
				treeTestSample{"drinks": false, "height": 5.4, "class": "teenager"},
				treeTestSample{"drinks": true, "height": 6.0, "class": "teenager"},
				treeTestSample{"drinks": true, "height": 6.0, "class": "adult"},
			},
			Attrs: []string{"height", "drinks"},
			Expected: &Tree{
				Attr: "height",
				NumSplit: &NumSplit{
					Threshold: (3.0 + 4.3) / 2.0,
					LessEqual: &Tree{
						Classification: map[interface{}]float64{"child": 1},
					},
					Greater: &Tree{
						Attr: "drinks",
						ValSplit: map[interface{}]*Tree{
							false: &Tree{
								Classification: map[interface{}]float64{"teenager": 1},
							},
							true: &Tree{
								Attr: "height",
								NumSplit: &NumSplit{
									Threshold: (6.0 + 5.5) / 2.0,
									LessEqual: &Tree{
										Classification: map[interface{}]float64{"adult": 1},
									},
									Greater: &Tree{
										Classification: map[interface{}]float64{
											"adult": 0.5, "teenager": 0.5,
										},
									},
								},
							},
						},
					},
				},
			},
		},
	}
	for i, test := range tests {
		test.Run(t, fmt.Sprintf("case %d", i))
	}
}
