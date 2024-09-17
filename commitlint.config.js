// These are the legal commit types:
//  https://github.com/angular/angular/blob/main/CONTRIBUTING.md#type
module.exports = {
    extends: ['@commitlint/config-angular'],
    rules: {
        'type-enum': [
            2,
            'always',
            [
                'build',
                'ci',
                'docs',
                'feat',
                'fix',
                'perf',
                'refactor',
                'test',
                'chore',
            ],
        ],
    },
}
