using Pkg
Pkg.add("PyCall")
using PyCall
np = pyimport("numpy")

println(np.zeros(8))