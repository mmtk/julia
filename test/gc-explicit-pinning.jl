# This file is a part of Julia. License is MIT: https://julialang.org/license

using Test

function check_if_pin_counts_are_initialized_to_zero()
    d = Dict() # Dummy object
    @test get_pin_count(d) == 0
    @test get_tpin_count(d) == 0
end

function check_if_pin_count_drops_to_zero()
    d = Dict() # Dummy object
    increment_pin_count!(d)
    @test get_pin_count(d) == 1
    @test get_tpin_count(d) == 0
    decrement_pin_count!(d)
    @test get_pin_count(d) == 0
    @test get_tpin_count(d) == 0
end
function check_if_tpin_count_drops_to_zero()
    d = Dict() # Dummy object
    increment_tpin_count!(d)
    @test get_pin_count(d) == 0
    @test get_tpin_count(d) == 1
    decrement_tpin_count!(d)
    @test get_pin_count(d) == 0
    @test get_tpin_count(d) == 0
end

function check_if_pin_count_stays_above_zero()
    d = Dict() # Dummy object
    increment_pin_count!(d)
    @test get_pin_count(d) == 1
    @test get_tpin_count(d) == 0
    increment_pin_count!(d)
    @test get_pin_count(d) == 2
    @test get_tpin_count(d) == 0
    decrement_pin_count!(d)
    @test get_pin_count(d) == 1
    @test get_tpin_count(d) == 0
end
function check_if_tpin_count_stays_above_zero()
    d = Dict() # Dummy object
    increment_tpin_count!(d)
    @test get_pin_count(d) == 0
    @test get_tpin_count(d) == 1
    increment_tpin_count!(d)
    @test get_pin_count(d) == 0
    @test get_tpin_count(d) == 2
    decrement_tpin_count!(d)
    @test get_pin_count(d) == 0
    @test get_tpin_count(d) == 1
end

mutable struct TreeNode
    value::Int
    left::Union{Nothing, TreeNode}
    right::Union{Nothing, TreeNode}
end
function create_tree(depth::Int)
    if depth == 0
        return nothing
    end
    left = create_tree(depth - 1)
    right = create_tree(depth - 1)
    return TreeNode(rand(1:100), left, right)
end
function dump_in_order_traversal_into_vector(node::Union{Nothing, TreeNode}, vec::Vector{TreeNode})
    if node === nothing
        return
    end
    dump_in_order_traversal_into_vector(node.left, vec)
    push!(vec, node)
    dump_in_order_traversal_into_vector(node.right, vec)
end

function check_pinning_on_recursive_structure()
    root = create_tree(5) # Create a tree with depth 5

    # Pin the root node
    increment_pin_count!(root)
    @test get_pin_count(root) == 1
    @test get_tpin_count(root) == 0

    # Run a couple of GCs to see if the root was not moved.
    # We approximate "object was not moved" by "type(tag) is still the same".
    for _ in 1:10
        GC.gc()
        @test typeof(root) == TreeNode
    end
end
function check_tpinning_on_recursive_structure()
    root = create_tree(5) # Create a tree with depth 5
    in_order_traversal = Vector{TreeNode}()
    dump_in_order_traversal_into_vector(root, in_order_traversal)

    # Transitively pin the root node
    increment_tpin_count!(root)
    @test get_pin_count(root) == 0
    @test get_tpin_count(root) == 1

    # Run a couple of GCs to see if the nodes in the tree were not moved.
    # We approximate "object was not moved" by "type(tag) is still the same".
    for _ in 1:10
        GC.gc()
        for node in in_order_traversal
            @test typeof(node) == TreeNode
        end
    end
end

function run_tests()
    check_if_pin_counts_are_initialized_to_zero()
    check_if_pin_count_drops_to_zero()
    check_if_tpin_count_drops_to_zero()
    check_if_pin_count_stays_above_zero()
    check_if_tpin_count_stays_above_zero()
    check_pinning_on_recursive_structure()
    check_tpinning_on_recursive_structure()
end
run_tests()
