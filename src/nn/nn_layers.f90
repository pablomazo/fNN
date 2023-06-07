! Heavily inspired by https://github.com/modern-fortran/neural-fortran/tree/main
module nn_layers
    use constants, only: dp
    use nn_activations, only: activation_func, linear
    implicit none
    type :: dense_layer
        integer :: input_size, output_size
        class(activation_func), allocatable :: activation
        real(dp), allocatable :: w(:,:), b(:), &
                                 z(:), & ! z = w * x + b
                                 output(:) ! output = activation(z)
        contains
            procedure :: forward
            !procedure :: set_parameters
            procedure :: init
    end type

    interface dense_layer
        module function build_layer(input_size, output_size, activation) result(layer)
            integer, intent(in) :: input_size, output_size
            class(activation_func), intent(in), optional :: activation
            type(dense_layer) :: layer
        end function
    end interface

    contains
    module function build_layer(input_size, output_size, activation) result(layer)
        integer, intent(in) :: input_size, output_size
        class(activation_func), intent(in), optional :: activation
        class(activation_func), allocatable :: activation_tmp
        type(dense_layer) :: layer
        layer % input_size = input_size
        layer % output_size = output_size
        if (present (activation)) then
            allocate(activation_tmp, source=activation)
        else
            allocate(activation_tmp, source=linear())
        end if
        allocate( layer % activation, source = activation_tmp )

    end function
    subroutine  init(self)
        class(dense_layer), intent(inout) :: self
        allocate(self % w(self % input_size, self % output_size))
        call random_number(self % w)
        allocate(self % b(self % output_size))
        self % b = 0_dp
        allocate(self % z(self % output_size))
        self % z = 0_dp
        allocate(self % output(self % output_size))
        self % output = 0_dp
    end subroutine

    subroutine forward(self, inp)
        class(dense_layer), intent(inout) :: self
        real(dp), intent(in) :: inp(:)
        integer :: i, j

        self % z = 0_dp
        do i=1, self % input_size
            do j=1, self % output_size
                self % z(j) = self % z(j) + self % w(i,j) * inp(i)
            end do
        end do
        self % z = self % z + self % b
        self%output = self % activation % eval(self % z)
    end subroutine
end module
