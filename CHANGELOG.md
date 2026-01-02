# Changelog

## [4.1.0](https://github.com/utiasDSL/crisp_gym/compare/v4.0.0...v4.1.0) (2025-11-21)


### Features

* add absolute actions environment ([5185584](https://github.com/utiasDSL/crisp_gym/commit/51855848ffe589a2407a6cda52f152694bff42d7))
* Add absolute actions to cartesian environment ([42eb715](https://github.com/utiasDSL/crisp_gym/commit/42eb715f9c4eadc833926180b921b8279f713885))
* Allow recordings in absolute pose ([b8cc64e](https://github.com/utiasDSL/crisp_gym/commit/b8cc64eea1de06508247a5922f0b45dbe287c920))


### Bug Fixes

* wrong warnings on gym version ([c9f21d8](https://github.com/utiasDSL/crisp_gym/commit/c9f21d82d83241efb847de76ebac7b0fd32b3b41))

## [4.0.0](https://github.com/utiasDSL/crisp_gym/compare/v3.2.3...v4.0.0) (2025-11-17)


### ⚠ BREAKING CHANGES

* Move environments to its own directory

### Features

* add orientation representation + observation to include to state ([08830e0](https://github.com/utiasDSL/crisp_gym/commit/08830e0cf32925c12e0cf0beb6bf49d7ab57731e))
* add reset for sensors ([0d15cf6](https://github.com/utiasDSL/crisp_gym/commit/0d15cf6acf90de6c2c4f67e029853578cb35eb0b))
* Include metadata in recording + add an option to randomize the home pose ([2becf29](https://github.com/utiasDSL/crisp_gym/commit/2becf29abb49df145debfa919cb3ac241f09a26b))
* make policy configurable with config file ([07ea87a](https://github.com/utiasDSL/crisp_gym/commit/07ea87a3b34f5f19b6e3da107f806d7da00b64e5))
* Make repo id mandatory ([2456e6c](https://github.com/utiasDSL/crisp_gym/commit/2456e6c53d08a23b4fd37bb73a6458f9c5b98144))
* Move environments to its own directory ([7b92cfd](https://github.com/utiasDSL/crisp_gym/commit/7b92cfd91a7ad3f86dea8e7e2579b243d17c5516))
* new configs ([1362f20](https://github.com/utiasDSL/crisp_gym/commit/1362f203b690265f91ab5a7cb30755559bd2bf5d))
* Policy interface to allow different inference types ([e575c4c](https://github.com/utiasDSL/crisp_gym/commit/e575c4c88f224c8f10f4126e8ca63c5e117cbe30))
* Push to hub is not default in deployment ([df35143](https://github.com/utiasDSL/crisp_gym/commit/df35143a03284561cf523034d00f54bef7ffcbf1))
* warn user of overrides in the policy and environment ([1dd0eb3](https://github.com/utiasDSL/crisp_gym/commit/1dd0eb3b1a53c290b3f10b2bd6daa1867f261f60))


### Bug Fixes

* a few small details ([32a7879](https://github.com/utiasDSL/crisp_gym/commit/32a7879355671320a482a20b3caad1c69680d674))
* evaluator was missing start_timer ([1dad235](https://github.com/utiasDSL/crisp_gym/commit/1dad2353478edf3c51a4f9c92c8a39bbebe472c3))
* pixi.toml lerobot install and add scripts to pyproject.toml ([e420997](https://github.com/utiasDSL/crisp_gym/commit/e4209970c5be9de2a0908cfc71310d999d26f623))
* removed unused import ([5a996b3](https://github.com/utiasDSL/crisp_gym/commit/5a996b37bffd3a3bff76178bda76cabcffaeebe2))

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
