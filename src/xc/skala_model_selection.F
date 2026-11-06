!--------------------------------------------------------------------------------------------------!
!   CP2K: A general program to perform molecular dynamics simulations                              !
!   Copyright 2000-2026 CP2K developers group <https://cp2k.org>                                   !
!                                                                                                  !
!   SPDX-License-Identifier: GPL-2.0-or-later                                                      !
!--------------------------------------------------------------------------------------------------!

! **************************************************************************************************
!> \brief Shared SKALA/GauXC section and model selection helpers
! **************************************************************************************************
MODULE skala_model_selection

   USE input_constants,                 ONLY: do_gpu_auto,&
                                              do_gpu_device,&
                                              do_gpu_host
   USE input_section_types,             ONLY: section_vals_get_subs_vals,&
                                              section_vals_get_subs_vals2,&
                                              section_vals_type,&
                                              section_vals_val_get
   USE kinds,                           ONLY: default_path_length
   USE string_utilities,                ONLY: uppercase
#include "../base/base_uses.f90"

   IMPLICIT NONE

   PRIVATE

   CHARACTER(len=*), PARAMETER, PRIVATE :: moduleN = 'skala_model_selection'

   PUBLIC :: find_functional_subsection, find_single_functional_subsection, &
             gauxc_model_none_selected, get_gauxc_section, get_skala_native_section, &
             resolve_skala_model_path, skala_gpu_mode_use_cuda, &
             xc_section_uses_gauxc, xc_section_uses_gauxc_model, &
             xc_section_uses_native_skala_grid, xc_section_uses_skala_model

CONTAINS

! **************************************************************************************************
!> \brief Return the XC_FUNCTIONAL subsection with the given name, if present.
!> \param xc_section ...
!> \param section_name ...
!> \return ...
! **************************************************************************************************
   FUNCTION find_functional_subsection(xc_section, section_name) RESULT(functional_section)
      TYPE(section_vals_type), INTENT(IN), POINTER       :: xc_section
      CHARACTER(len=*), INTENT(IN)                       :: section_name
      TYPE(section_vals_type), POINTER                   :: functional_section

      INTEGER                                            :: ifun
      TYPE(section_vals_type), POINTER                   :: functionals, xc_fun

      NULLIFY (functional_section)
      IF (.NOT. ASSOCIATED(xc_section)) RETURN

      functionals => section_vals_get_subs_vals(xc_section, "XC_FUNCTIONAL")
      IF (.NOT. ASSOCIATED(functionals)) RETURN

      ifun = 0
      DO
         ifun = ifun + 1
         xc_fun => section_vals_get_subs_vals2(functionals, i_section=ifun)
         IF (.NOT. ASSOCIATED(xc_fun)) EXIT
         IF (TRIM(xc_fun%section%name) == TRIM(section_name)) THEN
            functional_section => xc_fun
            EXIT
         END IF
      END DO

   END FUNCTION find_functional_subsection

! **************************************************************************************************
!> \brief Return the named functional subsection and the number of explicit XC_FUNCTIONAL subsections.
!> \param xc_section ...
!> \param section_name ...
!> \param functional_section ...
!> \param n_subsections ...
! **************************************************************************************************
   SUBROUTINE find_single_functional_subsection(xc_section, section_name, functional_section, &
                                                n_subsections)
      TYPE(section_vals_type), INTENT(IN), POINTER       :: xc_section
      CHARACTER(len=*), INTENT(IN)                       :: section_name
      TYPE(section_vals_type), POINTER                   :: functional_section
      INTEGER, INTENT(OUT)                               :: n_subsections

      INTEGER                                            :: ifun
      TYPE(section_vals_type), POINTER                   :: functionals, xc_fun

      NULLIFY (functional_section)
      n_subsections = 0
      IF (.NOT. ASSOCIATED(xc_section)) RETURN

      functionals => section_vals_get_subs_vals(xc_section, "XC_FUNCTIONAL")
      IF (.NOT. ASSOCIATED(functionals)) RETURN

      ifun = 0
      DO
         ifun = ifun + 1
         xc_fun => section_vals_get_subs_vals2(functionals, i_section=ifun)
         IF (.NOT. ASSOCIATED(xc_fun)) EXIT
         n_subsections = n_subsections + 1
         IF (TRIM(xc_fun%section%name) == TRIM(section_name)) functional_section => xc_fun
      END DO

   END SUBROUTINE find_single_functional_subsection

! **************************************************************************************************
!> \brief Return the CP2K-native SKALA functional subsection, if present.
!> \param xc_section ...
!> \return ...
! **************************************************************************************************
   FUNCTION get_skala_native_section(xc_section) RESULT(native_section)
      TYPE(section_vals_type), INTENT(IN), POINTER       :: xc_section
      TYPE(section_vals_type), POINTER                   :: native_section

      native_section => find_functional_subsection(xc_section, "SKALA_NATIVE")

   END FUNCTION get_skala_native_section

! **************************************************************************************************
!> \brief Return the first GAUXC functional subsection, if present.
!> \param xc_section ...
!> \return ...
! **************************************************************************************************
   FUNCTION get_gauxc_section(xc_section) RESULT(gauxc_section)
      TYPE(section_vals_type), INTENT(IN), POINTER       :: xc_section
      TYPE(section_vals_type), POINTER                   :: gauxc_section

      gauxc_section => find_functional_subsection(xc_section, "GAUXC")

   END FUNCTION get_gauxc_section

! **************************************************************************************************
!> \brief Return true if a GAUXC or SKALA_NATIVE subsection is present.
!> \param xc_section ...
!> \return ...
! **************************************************************************************************
   FUNCTION xc_section_uses_gauxc(xc_section) RESULT(uses_gauxc)
      TYPE(section_vals_type), INTENT(IN), POINTER       :: xc_section
      LOGICAL                                            :: uses_gauxc

      uses_gauxc = ASSOCIATED(get_gauxc_section(xc_section)) .OR. &
                   ASSOCIATED(get_skala_native_section(xc_section))

   END FUNCTION xc_section_uses_gauxc

! **************************************************************************************************
!> \brief Return true if the SKALA_NATIVE subsection is present.
!> \param xc_section ...
!> \return ...
! **************************************************************************************************
   FUNCTION xc_section_uses_native_skala_grid(xc_section) RESULT(uses_native_grid)
      TYPE(section_vals_type), INTENT(IN), POINTER       :: xc_section
      LOGICAL                                            :: uses_native_grid

      uses_native_grid = ASSOCIATED(get_skala_native_section(xc_section))

   END FUNCTION xc_section_uses_native_skala_grid

! **************************************************************************************************
!> \brief Return true if the GAUXC or SKALA_NATIVE subsection requests a model evaluation.
!> \param xc_section ...
!> \return ...
! **************************************************************************************************
   FUNCTION xc_section_uses_gauxc_model(xc_section) RESULT(uses_gauxc_model)
      TYPE(section_vals_type), INTENT(IN), POINTER       :: xc_section
      LOGICAL                                            :: uses_gauxc_model

      CHARACTER(len=default_path_length)                 :: model_name, xc_name
      TYPE(section_vals_type), POINTER                   :: gauxc_section, native_section

      uses_gauxc_model = .FALSE.
      native_section => get_skala_native_section(xc_section)
      IF (ASSOCIATED(native_section)) THEN
         uses_gauxc_model = .TRUE.
         RETURN
      END IF
      gauxc_section => get_gauxc_section(xc_section)
      IF (.NOT. ASSOCIATED(gauxc_section)) RETURN

      CALL section_vals_val_get(gauxc_section, "MODEL", c_val=model_name)
      CALL section_vals_val_get(gauxc_section, "FUNCTIONAL", c_val=xc_name)
      uses_gauxc_model = .NOT. model_key_is_none(model_name, xc_name)

   END FUNCTION xc_section_uses_gauxc_model

! **************************************************************************************************
!> \brief Return true if the XC section requests a SKALA model (GauXC or native/FTorch).
!> \param xc_section ...
!> \return ...
! **************************************************************************************************
   FUNCTION xc_section_uses_skala_model(xc_section) RESULT(uses_skala_model)
      TYPE(section_vals_type), INTENT(IN), POINTER       :: xc_section
      LOGICAL                                            :: uses_skala_model

      CHARACTER(len=default_path_length)                 :: model_key, model_name
      LOGICAL                                            :: section_active
      TYPE(section_vals_type), POINTER                   :: gauxc_section, native_section

      uses_skala_model = .FALSE.
      native_section => get_skala_native_section(xc_section)
      IF (ASSOCIATED(native_section)) THEN
         CALL section_vals_val_get(native_section, "_SECTION_PARAMETERS_", l_val=section_active)
         IF (section_active) THEN
            uses_skala_model = .TRUE.
            RETURN
         END IF
      END IF

      gauxc_section => get_gauxc_section(xc_section)
      IF (.NOT. ASSOCIATED(gauxc_section)) RETURN

      CALL section_vals_val_get(gauxc_section, "_SECTION_PARAMETERS_", l_val=section_active)
      IF (.NOT. section_active) RETURN

      CALL section_vals_val_get(gauxc_section, "MODEL", c_val=model_name)
      model_key = ADJUSTL(model_name)
      CALL uppercase(model_key)
      uses_skala_model = INDEX(TRIM(model_key), "SKALA") > 0

   END FUNCTION xc_section_uses_skala_model

! **************************************************************************************************
!> \brief Return true when a model key selects NONE or the functional itself.
!> \param model_name ...
!> \param xc_name ...
!> \return ...
! **************************************************************************************************
   FUNCTION model_key_is_none(model_name, xc_name) RESULT(none_selected)
      CHARACTER(len=*), INTENT(IN)                       :: model_name, xc_name
      LOGICAL                                            :: none_selected

      CHARACTER(len=default_path_length)                 :: model_key, xc_key

      model_key = ADJUSTL(model_name)
      xc_key = ADJUSTL(xc_name)
      CALL uppercase(model_key)
      CALL uppercase(xc_key)
      none_selected = (TRIM(model_key) == "" .OR. TRIM(model_key) == "NONE" .OR. &
                       TRIM(model_key) == TRIM(xc_key))

   END FUNCTION model_key_is_none

! **************************************************************************************************
!> \brief Return true when MODEL NONE (or no model distinct from FUNCTIONAL) is selected.
!> \param functional the GAUXC section
!> \brief Return true when MODEL NONE (or no model distinct from FUNCTIONAL) is selected.
!> \param none_selected ...
!> \return ...
! **************************************************************************************************
   FUNCTION gauxc_model_none_selected(functional) RESULT(none_selected)
      TYPE(section_vals_type), POINTER                   :: functional
      LOGICAL                                            :: none_selected

      CHARACTER(len=default_path_length)                 :: model_name, xc_fun_name

      CALL section_vals_val_get(functional, "MODEL", c_val=model_name)
      CALL section_vals_val_get(functional, "FUNCTIONAL", c_val=xc_fun_name)
      none_selected = model_key_is_none(model_name, xc_fun_name)

   END FUNCTION gauxc_model_none_selected

! **************************************************************************************************
!> \brief Resolve the USE_GPU mode to a logical CUDA decision.
!> \param gpu_mode ...
!> \param auto_cuda whether AUTO should select the GPU for this backend
!> \return ...
! **************************************************************************************************
   FUNCTION skala_gpu_mode_use_cuda(gpu_mode, auto_cuda) RESULT(use_cuda)
      INTEGER, INTENT(IN)                                :: gpu_mode
      LOGICAL, INTENT(IN)                                :: auto_cuda
      LOGICAL                                            :: use_cuda

      SELECT CASE (gpu_mode)
      CASE (do_gpu_auto)
         use_cuda = auto_cuda
      CASE (do_gpu_host)
         use_cuda = .FALSE.
      CASE (do_gpu_device)
         use_cuda = .TRUE.
      CASE DEFAULT
         CPABORT("Unknown SKALA execution-space value.")
      END SELECT

   END FUNCTION skala_gpu_mode_use_cuda

! **************************************************************************************************
!> \brief Resolve MODEL SKALA to the host or CUDA model path, or return a custom path unchanged.
!> \param model_name ...
!> \param use_cuda ...
!> \param model_path ...
! **************************************************************************************************
   SUBROUTINE resolve_skala_model_path(model_name, use_cuda, model_path)
      CHARACTER(len=*), INTENT(IN)                       :: model_name
      LOGICAL, INTENT(IN)                                :: use_cuda
      CHARACTER(len=*), INTENT(OUT)                      :: model_path

      CHARACTER(len=default_path_length)                 :: model_key
      INTEGER                                            :: env_status

      model_key = ADJUSTL(model_name)
      CALL uppercase(model_key)
      IF (TRIM(model_key) == "NONE" .OR. TRIM(model_key) == "") THEN
         CPABORT("SKALA requires MODEL SKALA or a TorchScript model path")
      ELSE IF (TRIM(model_key) == "SKALA") THEN
         IF (use_cuda) THEN
            CALL GET_ENVIRONMENT_VARIABLE("SKALA_CUDA_MODEL", model_path, STATUS=env_status)
            IF (env_status /= 0 .OR. LEN_TRIM(model_path) == 0) THEN
               CPABORT("GPU execution of SKALA requires SKALA_CUDA_MODEL")
            END IF
         ELSE
            CALL GET_ENVIRONMENT_VARIABLE("SKALA_MODEL", model_path, STATUS=env_status)
            IF (env_status /= 0 .OR. LEN_TRIM(model_path) == 0) THEN
               CPABORT("SKALA requires the SKALA_MODEL environment variable")
            END IF
         END IF
      ELSE
         model_path = model_name
      END IF

   END SUBROUTINE resolve_skala_model_path

END MODULE skala_model_selection
