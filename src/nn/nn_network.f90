! Heavily inspired by https://github.com/modern-fortran/neural-fortran/tree/main
module nn_network
    use nn_types, only: dp
    use nn_layers, only: dense_layer
    use nn_activations, only: activation_func
    implicit none
    public :: network

    type :: network
        type(dense_layer), allocatable :: layers(:)
        integer :: n
        contains

            procedure, private :: forward_i
            procedure, private :: predict_i
            generic :: forward => forward_i
            generic :: predict => predict_i
    end type

    interface network
        module function build_network(layers) result(net)
            type(dense_layer), intent(in) :: layers(:)
            type(network) :: net
        end function
    end interface

    interface forward
        module subroutine forward_i(self, input)
            use nn_types, only: dp
            class(network), intent(inout) :: self
            real(dp), intent(in) :: input(:)
        end subroutine
    end interface

    interface predict
        module function predict_i(self, input) result(output)
            class(network), intent(inout) :: self
            real(dp), intent(in) :: input(:)
            real(dp), allocatable :: output(:)
        end function
    end interface

    contains

    module function build_network(layers) result(net)
        type(dense_layer), intent(in) :: layers(:)
        type(network) :: net
        integer :: i

        net % layers = layers
        net % n = size(layers)

        do i=1,size(layers)
            call net % layers(i) % init()
        end do
    end function

    module subroutine forward_i(self, input)
        class(network), intent(inout) :: self
        real(dp), intent(in) :: input(:)
        integer :: n, i
        call self % layers(1) % forward(input)
        do i=2, self % n
            call self % layers(i) % forward(self % layers(i-1) % output)
        end do
    end subroutine

    module function predict_i(self, input) result(output)
        class(network), intent(inout) :: self
        real(dp), intent(in) :: input(:)
        real(dp), allocatable :: output(:)
        call self % forward(input)
        output = self % layers(self % n) % output
    end function
end module
