! Heavily inspired by https://github.com/modern-fortran/neural-fortran/tree/main
module nn_activations
    use constants, only: dp
    implicit none

    public :: activation_func
    public :: CELU
    public :: linear

    type, abstract :: activation_func
        contains
            procedure(eval_i), deferred :: eval_base
            procedure(eval_prime_i), deferred :: eval_base_prime

            generic :: eval => eval_base
            generic :: eval_prime => eval_base_prime
    end type

    abstract interface
        pure function eval_i(self, x) result(output)
            use constants, only: dp
            import :: activation_func
            class(activation_func), intent(in) :: self
            real(dp), intent(in) :: x(:)
            real(dp) :: output(size(x))
        end function
        
        pure function eval_prime_i(self, x) result(output)
            use constants, only: dp
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

    contains
    pure function eval_base_CELU(self, x) result(output)
        class(CELU), intent(in) :: self
        real(dp), intent(in) :: x(:)
        real(dp) :: output(size(x))

        output = self % alpha * (exp(x/self %alpha) - 1)
        where (output > 0._dp)
            output = 0._dp
        end where

        where (x > 0._dp)
            output = output + x
        end where
    end function
    pure function eval_base_prime_CELU(self, x) result(output)
        class(CELU), intent(in) :: self
        real(dp), intent(in) :: x(:)
        real(dp) :: output(size(x))

        output = 1._dp ! TODO: implement CELU prime
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
end module
