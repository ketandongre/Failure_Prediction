[GlobalParams]
  displacements = 'disp_x disp_y'
[]

[Mesh]
  file = 'platehole_circle_mesh.e'
[]

[Modules/TensorMechanics/Master]
  [all]
    add_variables = true
    # we added this in the first exercise problem
    strain = FINITE
    # enable the use of automatic differentiation objects
    use_automatic_differentiation = true
    generate_output = 'stress_xx stress_xy stress_yy stress_yx'
  []
[]

[BCs]
  [left_x]
    # we use the AD version of this boundary condition here...
    type = ADDirichletBC
    variable = disp_x
    boundary = left
    value = 0
  []
  [bottom_y]
    # ...and here
    type = ADDirichletBC
    variable = disp_y
    boundary = bottom
    value = 0
  []
  [Pressure]
    [right]
      boundary = right
      factor = -1
    []
  []
[]

[Materials]
  [elasticity]
    type = ADComputeIsotropicElasticityTensor
    youngs_modulus = 20
    poissons_ratio = 0.25
  []
  [stress]
    type = ADComputeFiniteStrainElasticStress
  []
[]

[Executioner]
  type = Transient
  # MOOSE automatically sets up SMP/full=true with NEWTON
  solve_type = NEWTON
  petsc_options_iname = '-pc_type'
  petsc_options_value = 'lu'
  end_time = 1
  dt = 1
[]

[Outputs]
  exodus = true
[]
