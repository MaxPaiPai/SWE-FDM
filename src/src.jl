start = time()

## define domain & physical quantities
x_range = 0.5E+6    ## 1/2 of the range of x
y_range = 0.5E+6    ## 1/2 of the range of y
depth   = 100       ## depth of water
density = 997       ## density of water
g       = 9.81      ## gravity constant

## define step size & grid construction
pointx = 100        ## number of grid pointer in x-axis
pointy = 100        ## number of grid pointer in y-axis
stepx  = pointx-1   ## total number of space step in x
stepy  = pointy-1   ## total number of space step in y

x = LinRange(-x_range, x_range, pointx)
y = LinRange(-y_range, y_range, pointy)

grid_x = zeros(pointx,pointy)
for i in 1:pointx
    grid_x[i,:] = x
end
grid_y = zeros(pointx,pointy)
for i in 1:pointy
    grid_y[i,:] = y
end
grid_y = grid_y'

initial_height = 10  ## the approximated height of inital water column

## preparation for discretization
delta_x = 2*x_range/stepx   ## step size of x
delta_y = 2*y_range/stepy   ## step size of y
delta_t = sqrt(0.5)*min(delta_x, delta_y)/sqrt(g*depth)    ## by cfl condition
t_final = 5000              ## total time step

## specify initial condition (2D Gaussian)
h_current = fill(initial_height, (pointx,pointy)) + exp(-((grid_x)^2/(100000^2) + (grid_y)^2/(100000^2)))

## construct discretization arrays
h_next      = zeros(pointx,pointy) 
u_current   = zeros(pointx,pointy) 
u_next      = zeros(pointx,pointy) 
v_current   = zeros(pointx,pointy) 
v_next      = zeros(pointx,pointy) 
x_momentum  = zeros(pointx,pointy) 
y_momentum  = zeros(pointx,pointy) 
height_list = Vector{Matrix{Float64}}()
height_list = push!(height_list, h_current);

for j in 1:t_final
    ## momentum equations
    u_next[1:pointx-1, :] .= u_current[1:pointx-1, :] - g*delta_t/delta_x*(h_current[2:pointx, :] - h_current[1:pointx-1, :])
    v_next[:, 1:pointy-1] .= v_current[:, 1:pointy-1] - g*delta_t/delta_y*(h_current[:, 2:pointy] - h_current[:, 1:pointy-1])

    ## set boundary velocity
    u_next[1, :]        .= 0.0
    u_next[pointx-1, :] .=  0.0
    v_next[:, 1]        .= 0.0
    v_next[:, pointy-1] .= 0.0

    ## continuity equation (lienarized continuity)
    x_momentum[1, :]        .= u_next[1, :] - (-u_current[1, :])
    x_momentum[2:pointx, :] .= u_next[2:pointx, :] - u_next[1:pointx-1, :]
    x_momentum[pointx, :]   .+= -u_current[pointx, :]
    x_momentum              .= x_momentum/delta_x

    y_momentum[:, 1]        .= v_next[:, 1] - (-v_current[:, 1])
    y_momentum[:, 2:pointy] .= v_next[:, 2:pointy] - v_next[:, 1:pointy-1]
    y_momentum[:, pointy]   .+= -v_current[:, pointy]
    y_momentum              .= y_momentum/delta_y   

    h_next .= h_current - delta_t*depth*(x_momentum + y_momentum)
    
    ## add sol. at current time
    height_list .= push!(height_list, h_next);

    ## prepare for next step
    u_current .= copy(u_next)
    v_current .= copy(v_next)
    h_current .= copy(h_next)
end

finish = time() - start
println("Number of Grid: ", pointx*pointy)
println("Execution Time: ", finish)
