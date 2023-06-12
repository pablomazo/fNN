! Heavily inspired by https://github.com/modern-fortran/neural-fortran/tree/main
module nn_layers
    use nn_types, only: dp
    use nn_activations, only: activation_func, linear
    implicit none
    type :: dense_layer
        integer :: input_size, output_size
        class(activation_func), allocatable :: activation
        real(dp), allocatable :: w(:,:), b(:), &
                                 z(:), & ! z = w * x + b
                                 output(:), & ! output = activation(z)
                                 grad(:,:) ! grad = d output / d input
        contains
            procedure :: forward
            procedure :: set_parameters
            procedure :: get_num_parameters
            procedure :: init
            procedure :: gradient
    end type

    interface dense_layer
        module function build_layer(input_size, output_size, activation) result(layer)
            integer, intent(in) :: input_size, output_size
            class(activation_func), intent(in), optional :: activation
            type(dense_layer) :: layer
        end function
    end interface

    interface
        module subroutine forward(self, input)
            use nn_types, only: dp
            class(dense_layer), intent(inout) :: self
            real(dp), intent(in) :: input(:)
        end subroutine

        module subroutine set_parameters(self, params)
            class(dense_layer), intent(inout) :: self
            real(dp), intent(in) :: params(:)
        end subroutine

        module function get_num_parameters(self) result(params)
            class(dense_layer), intent(inout) :: self
            integer :: params
        end function

        module subroutine gradient(self, grad)
            use nn_types, only: dp
            class(dense_layer), intent(inout) :: self
            real(dp), intent(in) :: grad(:,:)
        end subroutine
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
    module subroutine  init(self)
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

    module subroutine forward(self, inp)
        class(dense_layer), intent(inout) :: self
        real(dp), intent(in) :: inp(:)
        integer :: i, j

        self % z = matmul(inp, self % w) + self % b
        self%output = self % activation % eval(self % z)
    end subroutine

    module subroutine set_parameters(self, params)
        class(dense_layer), intent(inout) :: self
        real(dp), intent(in) :: params(:)
        integer :: iw
        iw = self % input_size * self % output_size
        self % w = reshape(params(:iw), [self % input_size, self % output_size])
        self % b = params(iw+1:)
    end subroutine

    module function get_num_parameters(self) result(params)
        class(dense_layer), intent(inout) :: self
        integer :: params
        params = self % input_size * self % output_size + self % output_size
    end function

    module subroutine gradient(self, grad)
        class(dense_layer), intent(inout) :: self
        real(dp), intent(in) :: grad(:,:)
        real(dp) :: aux(self % input_size, self % output_size), vec(self % output_size)
        integer :: s
        s = size(grad,2)
        if (.not. allocated(self % grad)) allocate(self % grad(self % input_size, s))
        vec = self % activation % eval_prime(self % z)
        aux = self % w * spread(vec, 1, self % input_size)
        if (s == 1) then
            vec = grad(:,1)
            self % grad(:,1) = matmul(aux, vec)
        else
        self % grad = matmul(aux, grad)
        end if
    end subroutine
end module
