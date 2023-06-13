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

            procedure :: set_parameters
            procedure :: get_num_parameters
            procedure, private :: forward_i
            procedure, private :: predict_i
            procedure, private :: gradient_i
            generic :: forward => forward_i
            generic :: predict => predict_i
            generic :: gradient => gradient_i
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

    interface gradient
        module subroutine gradient_i(self)
            class(network), intent(inout) :: self
        end subroutine
    end interface

    interface
        module subroutine set_parameters(self, params)
            class(network), intent(inout) :: self
            real(dp), intent(in) :: params(:)
        end subroutine

        module function get_num_parameters(self) result(params)
            class(network), intent(inout) :: self
            integer :: params
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

    module subroutine set_parameters(self, params)
        class(network), intent(inout) :: self
        real(dp), intent(in) :: params(:)
        integer :: nstart, nend, i

        nstart = 1
        do i=1, self % n
            nend = nstart + self % layers(i) % get_num_parameters() - 1
            call self % layers(i) % set_parameters(params(nstart:nend))
            nstart = nend + 1
        end do
    end subroutine

    module function get_num_parameters(self) result(num)
        class(network), intent(inout) :: self
        integer :: num, i

        num = 0
        do i=1, self % n
            num = num + self % layers(i) % get_num_parameters()
        end do
    end function

    module subroutine gradient_i(self)
        class(network), intent(inout) :: self
        integer :: ilayer
        call self % layers(self % n) % gradient([1d0])
        do ilayer=self % n-1,1,-1
            call self % layers(ilayer) % gradient(self % layers(ilayer+1) % grad)
        end do
    end subroutine
end module
