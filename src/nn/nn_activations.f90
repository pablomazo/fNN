! Heavily inspired by https://github.com/modern-fortran/neural-fortran/tree/main
module nn_activations
    use nn_types, only: dp
    implicit none

    public :: activation_func
    public :: CELU, &
        linear, &
        Sigmoid, &
        Tanhf

    type, abstract :: activation_func
        contains
            procedure(eval_i), deferred :: eval_base
            procedure(eval_prime_i), deferred :: eval_base_prime

            generic :: eval => eval_base
            generic :: eval_prime => eval_base_prime
    end type

    abstract interface
        pure function eval_i(self, x) result(output)
            use nn_types, only: dp
            import :: activation_func
            class(activation_func), intent(in) :: self
            real(dp), intent(in) :: x(:)
            real(dp) :: output(size(x))
        end function
        
        pure function eval_prime_i(self, x) result(output)
            use nn_types, only: dp
            import :: activation_func
            class(activation_func), intent(in) :: self
            real(dp), intent(in) :: x(:)
            real(dp) :: output(size(x))
        end function
    end interface

    type, extends(activation_func) :: CELU
        real(dp) :: alpha = 1.0_dp
        contains
            procedure :: eval_base => eval_base_CELU
            procedure :: eval_base_prime => eval_base_prime_CELU
    end type

    type, extends(activation_func) :: linear
        contains
            procedure :: eval_base => eval_base_linear
            procedure :: eval_base_prime => eval_base_prime_linear
    end type

    type, extends(activation_func) :: Sigmoid
        contains
            procedure :: eval_base => eval_base_sigmoid
            procedure :: eval_base_prime => eval_base_prime_sigmoid
    end type

    type, extends(activation_func) :: Tanhf
        contains
            procedure :: eval_base => eval_base_tanh
            procedure :: eval_base_prime => eval_base_prime_tanh
    end type
    contains
    pure function eval_base_CELU(self, x) result(output)
        class(CELU), intent(in) :: self
        real(dp), intent(in) :: x(:)
        real(dp) :: output(size(x))

        where (x >= 0._dp)
            output = x
        else where
            output = self % alpha * (exp(x/self %alpha) - 1)
        end where
    end function
    pure function eval_base_prime_CELU(self, x) result(output)
        class(CELU), intent(in) :: self
        real(dp), intent(in) :: x(:)
        real(dp) :: output(size(x))

        where (x >= 0._dp)
            output = 1._dp
        else where
            output = exp(x/self %alpha)
        end where
    end function

    pure function eval_base_linear(self, x) result(output)
        class(linear), intent(in) :: self
        real(dp), intent(in) :: x(:)
        real(dp) :: output(size(x))

        output = x
    end function

    pure function eval_base_prime_linear(self, x) result(output)
        class(linear), intent(in) :: self
        real(dp), intent(in) :: x(:)
        real(dp) :: output(size(x))

        output = 1._dp
    end function

    pure function eval_base_sigmoid(self, x) result(output)
        class(Sigmoid), intent(in) :: self
        real(dp), intent(in) :: x(:)
        real(dp) :: output(size(x))

        output = 1._dp / (1._dp + exp(-x))
    end function

    pure function eval_base_prime_sigmoid(self, x) result(output)
        class(Sigmoid), intent(in) :: self
        real(dp), intent(in) :: x(:)
        real(dp) :: output(size(x)), e(size(x))

        e = exp(-x)
        output = e / (1._dp + e)**2
    end function

    pure function eval_base_tanh(self, x) result(output)
        class(Tanhf), intent(in) :: self
        real(dp), intent(in) :: x(:)
        real(dp) :: output(size(x))

        output = tanh(x)
    end function

    pure function eval_base_prime_tanh(self, x) result(output)
        class(Tanhf), intent(in) :: self
        real(dp), intent(in) :: x(:)
        real(dp) :: output(size(x)), e(size(x))

        output = 1._dp - tanh(x)**2
    end function
end module
