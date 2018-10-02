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


#include "panoc.hpp"

#include "casadi/core/casadi_misc.hpp"
#include "casadi/core/calculus.hpp"

extern "C" {
#include "optimizer.h"
}

using namespace std;
namespace casadi {

  extern "C"
  int CASADI_NLPSOL_PANOC_EXPORT
      casadi_register_nlpsol_panoc(Nlpsol::Plugin* plugin) {
    plugin->creator = Panoc::creator;
    plugin->name = "panoc";
    plugin->doc = Panoc::meta_doc.c_str();
    plugin->version = CASADI_VERSION;
    plugin->options = &Panoc::options_;
    return 0;
  }

  extern "C"
  void CASADI_NLPSOL_PANOC_EXPORT casadi_load_nlpsol_panoc() {
    Nlpsol::registerPlugin(casadi_register_nlpsol_panoc);
  }

  Panoc::Panoc(const std::string& name, const Function& nlp)
    : Nlpsol(name, nlp) {
  }

  Panoc::~Panoc() {
    uout() << "Solver class destroyed" << std::endl;
  }

/* types : OT_INT,OT_DOUBLE,OT_STRING */
  Options Panoc::options_
  = {{&Nlpsol::options_},
     {{"outer_iterations",
       {OT_INT,
        "Maximum number of loops solving the lagrangian."}},
      {"inter_iterations",
       {OT_INT,
        "Maximum number of loops solving with PANOC."}},
      {"tolerance",
       {OT_DOUBLE,
        "Tolerance of PANOC solver"}},
      {"constraints_weight",
       {OT_DOUBLE,
        "weight of the contraints"}},
     }
  };

  void Panoc::init(const Dict& opts) {
    // Call the init method of the base class
    Nlpsol::init(opts);

    uout() << "Solver class init" << std::endl;

    // Avoid the superclass fixing lam_x
    bound_consistency_ = false;

    // Default options
    outer_iterations_=10;
    inner_iterations_=300;
    tolerance_=0.001;
    constraints_weight_= 1; //std::vector<double>(nx_, 1.);
    buffer_size_=15; 

    // Read user options
    for (auto&& op : opts) {
      if (op.first=="outer_iterations") {
        outer_iterations_ = op.second;
      } else if (op.first=="inner_iterations") {
        inner_iterations_ = op.second;
      } else if (op.first=="tolerance") {
        tolerance_ = op.second;
      } else if (op.first=="constraints_weight") {
        constraints_weight_ = op.second;
      } else if (op.first=="buffer_size") {
        buffer_size_ = op.second;
      }
    }

    //casadi_assert(constraints_weight_.size()==nx_, "Dimension mismatch constraints_weight");

    uout() << "outer_iterations: " << outer_iterations_ << std::endl;

    // First order derivative information
    Function nlp_jac_fg = create_function("nlp_jac_fg", {"x", "p"},
      {"f", "grad:f:x"});


    create_function("nlp_g", {"x", "p"}, {"g"});
    create_function("nlp_f", {"x", "p"}, {"f"});
    create_function("nlp_fwd_g", {"x", "p","fwd:x"},{"fwd:g"});

    // Allocate space in the work vector for the (dense) objective gradient
    alloc_w(nx_, true);

  }

  void Panoc::set_work(void* mem, const double**& arg, double**& res,
                                casadi_int*& iw, double*& w) const {
    auto m = static_cast<PanocMemory*>(mem);

    // Set work in base classes
    Nlpsol::set_work(mem, arg, res, iw, w);

    // Set gf pointer to the first available location in the work vector
    m->gf = w; w += nx_;

  }

  extern "C" {

    static struct MyData {
      const void* object;
      void* mem;
    } userdata;

    int constraints(const real_t* x, real_t* out) {
      uout() << "constraints " << &userdata << std::endl;
      auto m = static_cast<PanocMemory*>(userdata.mem);
      auto o = static_cast<const Panoc*>(userdata.object);

      // Set x,p inputs
      m->arg[0] = x;
      m->arg[1] = m->p;

      // Set g outputs
      m->res[0] = out;

      // Compute
      o->calc_function(m, "nlp_g");
      return SUCCESS;
    }

    static int constraints_forward_diff(const real_t* x,const real_t* y,real_t* out){
      uout() << "constraints diff" << &userdata << std::endl;

      auto m = static_cast<PanocMemory*>(userdata.mem);
      auto o = static_cast<const Panoc*>(userdata.object);

      // Set x,p inputs
      m->arg[0] = x;
      m->arg[1] = m->p;
      m->arg[2] = y; 

      // Set g outputs
      m->res[0] = out;

      // Compute
      o->calc_function(m, "nlp_fwd_g");

      return SUCCESS;
    }

    double cost_gradient(const double* input, double* gradient){
      uout() << "cost gradient " << &userdata << std::endl;
      auto m = static_cast<PanocMemory*>(userdata.mem);
      auto o = static_cast<const Panoc*>(userdata.object);
      double cost;

      m->arg[0] = input;
      m->arg[1] = m->p;

      m->res[0] = &cost; 
      m->res[1] = gradient;

      o->calc_function(m, "nlp_jac_fg");

      return cost;
    }

  }

  int Panoc::solve(void* mem) const {
    auto m = static_cast<PanocMemory*>(mem);

    userdata.object = this;
    userdata.mem = mem; 

    uout() << "Solve"  << std::endl;

    uout() << "Initial guess for 'x':" <<
      std::vector<double>(m->x, m->x+nx_) << std::endl;
    uout() << "Initial guess for 'lam_x':" <<
      std::vector<double>(m->lam_x, m->lam_x+nx_) << std::endl;
    uout() << "Initial guess for 'lam_g':" <<
      std::vector<double>(m->lam_g, m->lam_g+ng_) << std::endl;
    uout() << "Parameter values:" <<
      std::vector<double>(m->p, m->p+np_) << std::endl;

    struct optimizer_extended_problem extended_problem;
    extended_problem.problem.cost_gradient_function=cost_gradient;
    extended_problem.problem.dimension=nx_;

    extended_problem.lower_bounds_constraints=m->lbg;
    extended_problem.upper_bounds_constraints=m->ubg;
    extended_problem.constraints=constraints;
    extended_problem.number_of_constraints=ng_;
    extended_problem.constraints_forwad_diff=constraints_forward_diff;
    
    extended_problem.max_loops=outer_iterations_;
    extended_problem.problem.solver_params.tolerance=tolerance_;
    extended_problem.problem.solver_params.buffer_size=buffer_size_;
    extended_problem.problem.solver_params.max_interations=inner_iterations_;

    uout() << &userdata << std::endl;

    for(int i=0; i<1; i++){
     optimizer_init_extended_box(&extended_problem,m->lbx[i],m->ubx[i],constraints_weight_);
    }

    uout() << &userdata << std::endl;
    m->iter_count = solve_extended_problem(m->x);



    m->arg[0] = m->x;
    m->arg[1] = m->p;

    m->res[0] = &m->f;
    calc_function(m, "nlp_f");


    return 0;
  }

  Dict Panoc::get_stats(void* mem) const {
    Dict stats = Nlpsol::get_stats(mem);
    auto m = static_cast<PanocMemory*>(mem);
    stats["iter_count"] = m->iter_count;
    return stats;
  }
} // namespace casadi
