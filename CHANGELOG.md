# Changelog

## [3.2.3](https://github.com/utiasDSL/crisp_gym/compare/v3.2.2...v3.2.3) (2025-10-15)


### Bug Fixes

* ruff check ([27687f7](https://github.com/utiasDSL/crisp_gym/commit/27687f7434aa8c67920bee288429cc2e3fe95d47))

## [3.2.2](https://github.com/utiasDSL/crisp_gym/compare/v3.2.1...v3.2.2) (2025-10-15)


### Bug Fixes

* add crisp_gym config to config paths ([1e738b5](https://github.com/utiasDSL/crisp_gym/commit/1e738b530515e22c535532fb6fe235ba59bb0286))
* Deploy script get_features function had wrong arguments ([873531d](https://github.com/utiasDSL/crisp_gym/commit/873531d0e2aaec0816fe2bbe5f18a41497067bc1))
* gripper action needs clipping ([39df148](https://github.com/utiasDSL/crisp_gym/commit/39df1481e933110060031d0e43b7ef6cb2da8992))

## [3.2.1](https://github.com/utiasDSL/crisp_gym/compare/v3.2.0...v3.2.1) (2025-10-14)


### Bug Fixes

* Add deprecation warning for old parameters ([a339dfe](https://github.com/utiasDSL/crisp_gym/commit/a339dfe3997855e41b4d31d52e3f623450d78eee))
* gripper mode not properly initialized ([a7528e9](https://github.com/utiasDSL/crisp_gym/commit/a7528e9e7ee0816e208d096a8a83b649d5d29e78))

## [3.2.0](https://github.com/utiasDSL/crisp_gym/compare/v3.1.0...v3.2.0) (2025-10-13)


### Features

* Add different gripper control modes([#28](https://github.com/utiasDSL/crisp_gym/issues/28)) ([9d0e2b3](https://github.com/utiasDSL/crisp_gym/commit/9d0e2b312bdb839d5b95c429b11957e3c0d3b3f6))

## [3.1.0](https://github.com/utiasDSL/crisp_gym/compare/v3.0.0...v3.1.0) (2025-10-09)


### Features

* Recording manager config (cleaning it up) + new config examples for environments ([#26](https://github.com/utiasDSL/crisp_gym/issues/26)) ([64b8a9a](https://github.com/utiasDSL/crisp_gym/commit/64b8a9a2d42c0437b597754f2544147d66ed936f))

## [3.0.0](https://github.com/utiasDSL/crisp_gym/compare/v2.0.3...v3.0.0) (2025-09-29)


### ⚠ BREAKING CHANGES

* New observation state variable including all states in one

### Features

* add streamed teleop to script ([466fa42](https://github.com/utiasDSL/crisp_gym/commit/466fa42a5224d2a51ff6220f3c2baca01b9b1c5f))
* New observation state variable including all states in one ([bd94e29](https://github.com/utiasDSL/crisp_gym/commit/bd94e297d332652ea3d75894cbce2e4cfa0b9663))

## [2.0.3](https://github.com/utiasDSL/crisp_gym/compare/v2.0.2...v2.0.3) (2025-08-22)


### Bug Fixes

* hotfix variable name from ctrl_type to control_type ([547ae30](https://github.com/utiasDSL/crisp_gym/commit/547ae306a962da4635bfe4cfbcbde748c1b963b2))

## [2.0.2](https://github.com/utiasDSL/crisp_gym/compare/v2.0.1...v2.0.2) (2025-08-20)


### Bug Fixes

* release type was wrong for please release bot ([977ff20](https://github.com/utiasDSL/crisp_gym/commit/977ff20ef617252727a6c0895e43b9d029b525b2))

## [2.0.1](https://github.com/utiasDSL/crisp_gym/compare/v2.0.0...v2.0.1) (2025-08-20)


### Bug Fixes

* hotfix logging in record in new version ([b4ed37e](https://github.com/utiasDSL/crisp_gym/commit/b4ed37ed5bcee749e42c17dd6723f64d681ea5b7))

## [2.0.0](https://github.com/utiasDSL/crisp_gym/compare/v1.0.0...v2.0.0) (2025-08-19)


### ⚠ BREAKING CHANGES

* new way of configuring environments - multiple sources of config are now allowed. User can add config paths to env variable CRISP_CONFIG_PATH and separate each path with a column and then check that all paths are properly detected with the script: `pixi run python scripts/check_config.py`

### Features

* add config path from installed crisp_py pypi package directly ([6a5b302](https://github.com/utiasDSL/crisp_gym/commit/6a5b302e1701b4ffa46e4d062ad8004aca13e954))
* add multiple config paths to your env variable to allow for more flexibility ([6866627](https://github.com/utiasDSL/crisp_gym/commit/6866627deb963d37948fa09bd90977f028e54a27))
* add video saving to the lerobot recordings ([649d894](https://github.com/utiasDSL/crisp_gym/commit/649d89478adcce15873e00ac434e374980d52659))
* better configs for environment ([929f315](https://github.com/utiasDSL/crisp_gym/commit/929f3158b0a0a12194ab8a1b24ad2ddf0f368bbb))
* better factory classes + removing some robot specific data ([c772d19](https://github.com/utiasDSL/crisp_gym/commit/c772d194e02353423b7ca95496b8da88e33babfd))
* factory functions for teleop and manipulator env ([128f3f5](https://github.com/utiasDSL/crisp_gym/commit/128f3f59befb51220d8b9daf79835a291e3a2847))
* joint size not hardcoded anymore in the manipulator envrionments ([74714e0](https://github.com/utiasDSL/crisp_gym/commit/74714e0cedefc66735eec1c40ddb5277756eb81f))
* new way of configuring environments - multiple sources of config are now allowed. User can add config paths to env variable CRISP_CONFIG_PATH and separate each path with a column and then check that all paths are properly detected with the script: `pixi run python scripts/check_config.py` ([b7ae42c](https://github.com/utiasDSL/crisp_gym/commit/b7ae42c9a4e97e17670bcf6eb08819c8a360125b))
* recording manager factory ([3f2505c](https://github.com/utiasDSL/crisp_gym/commit/3f2505c9d594d9bb37fb20e4ec01c4dafc441e1a))
* upgraded to crisp-python&gt;=1.4.0 ([c229985](https://github.com/utiasDSL/crisp_gym/commit/c22998582b18070bbeccb2a2d3ed2c287399a483))
* yaml creation of enviroment ([48c9dc3](https://github.com/utiasDSL/crisp_gym/commit/48c9dc38895c00cff5f214cc1758a3c568b7bda9))

## 1.0.0 (2025-08-10)


### Bug Fixes

* lerobot dependency import + process to play sound ([5715105](https://github.com/utiasDSL/crisp_gym/commit/5715105166aa00d81a08948e5dd23c688d6a5d34))
