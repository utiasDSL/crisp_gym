# Changelog

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
