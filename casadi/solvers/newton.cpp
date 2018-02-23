/*
 *    This file is part of CasADi.
 *
 *    CasADi -- A symbolic framework for dynamic optimization.
 *    Copyright (C) 2010-2014 Joel Andersson, Joris Gillis, Moritz Diehl,
 *                            K.U. Leuven. All rights reserved.
 *    Copyright (C) 2011-2014 Greg Horn
 *
 *    CasADi is free software; you can redistribute it and/or
 *    modify it under the terms of the GNU Lesser General Public
 *    License as published by the Free Software Foundation; either
 *    version 3 of the License, or (at your option) any later version.
 *
 *    CasADi is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 *    Lesser General Public License for more details.
 *
 *    You should have received a copy of the GNU Lesser General Public
 *    License along with CasADi; if not, write to the Free Software
 *    Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
 *
 */


#include "newton.hpp"
#include <iomanip>
#include "../core/linsol_internal.hpp"

using namespace std;
namespace casadi {

  extern "C"
  int CASADI_ROOTFINDER_NEWTON_EXPORT
  casadi_register_rootfinder_newton(Rootfinder::Plugin* plugin) {
    plugin->creator = Newton::creator;
    plugin->name = "newton";
    plugin->doc = Newton::meta_doc.c_str();
    plugin->version = CASADI_VERSION;
    plugin->options = &Newton::options_;
    return 0;
  }

  extern "C"
  void CASADI_ROOTFINDER_NEWTON_EXPORT casadi_load_rootfinder_newton() {
    Rootfinder::registerPlugin(casadi_register_rootfinder_newton);
  }

  Newton::Newton(const std::string& name, const Function& f)
    : Rootfinder(name, f) {
  }

  Newton::~Newton() {
    clear_mem();
  }

  Options Newton::options_
  = {{&Rootfinder::options_},
     {{"abstol",
       {OT_DOUBLE,
        "Stopping criterion tolerance on ||g||__inf)"}},
      {"abstolStep",
       {OT_DOUBLE,
        "Stopping criterion tolerance on step size"}},
      {"max_iter",
       {OT_INT,
        "Maximum number of Newton iterations to perform before returning."}},
      {"print_iteration",
       {OT_BOOL,
        "Print information about each iteration"}}
     }
  };

  void Newton::init(const Dict& opts) {

    // Call the base class initializer
    Rootfinder::init(opts);

    // Default options
    max_iter_ = 1000;
    abstol_ = 1e-12;
    abstolStep_ = 1e-12;
    print_iteration_ = false;

    // Read options
    for (auto&& op : opts) {
      if (op.first=="max_iter") {
        max_iter_ = op.second;
      } else if (op.first=="abstol") {
        abstol_ = op.second;
      } else if (op.first=="abstolStep") {
        abstolStep_ = op.second;
      } else if (op.first=="print_iteration") {
        print_iteration_ = op.second;
      }
    }

    casadi_assert(oracle_.n_in()>0,
                          "Newton: the supplied f must have at least one input.");
    casadi_assert(!linsol_.is_null(),
                          "Newton::init: linear_solver must be supplied");

    // Allocate memory
    alloc_w(n_, true); // x
    alloc_w(n_, true); // F
    alloc_w(sp_jac_.nnz(), true); // J
  }

 void Newton::set_work(void* mem, const double**& arg, double**& res,
                       casadi_int*& iw, double*& w) const {
     Rootfinder::set_work(mem, arg, res, iw, w);
     auto m = static_cast<NewtonMemory*>(mem);
     m->x = w; w += n_;
     m->f = w; w += n_;
     m->jac = w; w += sp_jac_.nnz();
  }

  void iter_print_fun(casadi_int iter, double abstol, double abstol_step) {
    std::ostream& stream = uout();
    // Only print iteration header once in a while
    if (iter % 10==0) {
      stream << setw(5) << "iter";
      stream << setw(10) << "res";
      stream << setw(10) << "step";
      stream << std::endl;
      stream.unsetf(std::ios::floatfield);
    }

    // Print iteration information
    stream << setw(5) << iter;
    stream << setw(10) << scientific << setprecision(2) << abstol;
    stream << setw(10) << scientific << setprecision(2) << abstol_step;

    stream << fixed;
    stream << std::endl;
    stream.unsetf(std::ios::floatfield);
  }

  int Newton::solve(void* mem) const {
    auto m = static_cast<NewtonMemory*>(mem);

    auto fun_eval = ([&](void *, void*, void*, void*, void*) {
      calc_function(m, "jac_f_z");
    });
    auto solve_step = ([&](double* jac, double* f) {
      linsol_.nfact(m->jac);
      linsol_.solve(m->jac, m->f, 1, false);
    });
    typedef void (*iter_print_type)(casadi_int, double, double);
    iter_print_type iter_print = nullptr;
    if (print_iteration_) iter_print = iter_print_fun;
    casadi_int status;

    int ret = casadi_newton(fun_eval, solve_step, iter_print,
      m->iarg, m->ires,
      m->arg, m->res, m->iw, m->w,
      m->x, m->f, m->jac,
      n_, max_iter_, abstol_, abstolStep_,
      &m->iter, &status,
      iin_, iout_, n_in_, n_out_);

    switch (status) {
      case -1:
        if (verbose_)
          casadi_message("Max iterations reached.");
        m->return_status = "max_iteration_reached";
        break;
      case 0:
        if (verbose_)
          casadi_message("Accepted residual tolerance: " + str(casadi_norm_inf(n_, m->f)));
        m->return_status = "success";
        break;
      case 1:
        if (verbose_)
          casadi_message("Accepted step tolerance: " + str(casadi_norm_inf(n_, m->f)));
        m->return_status = "success";
        break;
      default:
        casadi_error("Unknown return code");
    }

    if (verbose_) casadi_message("Newton algorithm took " + str(m->iter) + " steps");
    return ret;
  }

  void Newton::codegen_body(CodeGenerator& g) const {
    g.add_auxiliary(CodeGenerator::AUX_COPY);
    g.add_auxiliary(CodeGenerator::AUX_AXPY);
    g.add_auxiliary(CodeGenerator::AUX_NORM_INF);
    g.add_auxiliary(CodeGenerator::AUX_NEWTON);

    casadi_int w_offset = 0;
    std::string x = "w"; w_offset+=n_;
    std::string f = "w+"+str(w_offset); w_offset+=n_;
    std::string jac = "w+"+str(w_offset); w_offset+=sp_jac_.nnz();
    std::string w = "w+"+str(w_offset);

    std::string jac_f_z = g.add_dependency(get_function("jac_f_z"));

    g << "casadi_int status, iter;\n";
    g << "return casadi_newton(" << jac_f_z << ", ";
    g << g.shorthand("rfp_linsol_"  + codegen_name(g));
    g << ", 0";
    g << ", arg, res, arg+" + str(n_in_) + ", res+"+str(n_out_) +  ", iw, " + w;
    g << ", " + x + ", " + f + ", " + jac;
    g << ", " << n_ << ", " << max_iter_ << ", " << abstol_ << ", " << abstolStep_;
    g << ", &iter, &status";
    g << ", " << iin_ << ", " << iout_ << ", " << n_in_ << ", " << n_out_;
    g << ");\n";
  }

  void Newton::codegen_declarations(CodeGenerator& g) const {
    g.add_dependency(get_function("jac_f_z"));

    string solver = g.shorthand("rfp_linsol_" + codegen_name(g));

    g << "void " << solver << "(casadi_real* jac, casadi_real* f) {\n";
    linsol_->generate(g, "jac", "f", 1, false);
    g << "}\n\n";
  }

  int Newton::init_mem(void* mem) const {
    if (Rootfinder::init_mem(mem)) return 1;
    auto m = static_cast<NewtonMemory*>(mem);
    m->return_status = 0;
    m->iter = 0;
    return 0;
  }

} // namespace casadi
