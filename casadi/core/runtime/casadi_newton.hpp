// NOLINT(legal/copyright)
// C-REPLACE "solve_step_type solve_step" "void (*solve_step)(casadi_real*, casadi_real*)"
// C-REPLACE "fun_eval_type jac_f_z" "int (*jac_f_z)(const casadi_real**, casadi_real**, casadi_int*, casadi_real*, void*)" // NOLINT(whitespace/line_length)
// SYMBOL "newton"
template<typename T1, typename fun_eval_type, typename solve_step_type>
int casadi_newton(fun_eval_type jac_f_z,
  solve_step_type solve_step,
  void (*iter_print)(casadi_int, T1, T1),
  const T1** iarg, T1** ires,
  const T1** arg, T1** res, casadi_int* iw, T1* w,
  T1* x, T1* f, T1* jac,
  casadi_int n, casadi_int max_iter, T1 abstol, T1 abstol_step,
  casadi_int* iter, casadi_int* status,
  casadi_int iin, casadi_int iout, casadi_int n_in, casadi_int n_out) {
  // Returns 0 when successful, 1 otherwise
  // Status: 0 residual tolerance met, 1 step tolerance met, -1 max iterations reached.

  casadi_int i;
  T1 norm_g;
  int ret;
  *iter = 0;
  ret = 1;

  // Get the initial guess
  casadi_copy(iarg[iin], n, x);

  while (1) {
    // Break if maximum number of iterations already reached
    if (*iter >= max_iter) {
      *status = -1;
      ret = 1; break;
    }
    (*iter)++;

    // Use x to evaluate J
    for (i=0;i<n_in;++i) arg[i] = iarg[i];
    arg[iin] = x;
    res[0] = jac;
    for (i=0;i<n_out;++i) res[i+1] = ires[i];
    res[1+iout] = f;
    jac_f_z(arg, res, iw, w, 0);

    norm_g = casadi_norm_inf(n, f);

    // Check convergence
    if (norm_g <= abstol) {
      *status = 0;
      ret = 0; break;
    }

    // Factorize the linear solver with J
    solve_step(jac, f);

    // Check convergence
    if (casadi_norm_inf(n, f) <= abstol_step) {
      *status = 1;
      ret = 0; break;
    }

    if (iter_print) iter_print(*iter, norm_g, casadi_norm_inf(n, f));

    // Update Xk+1 = Xk - J^(-1) F
    casadi_axpy(n, -1., f, x);
  }

  casadi_copy(x, n, ires[iout]);
  return ret;
}
