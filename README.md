# README

This README is work in progress. To be updated accordingly.

## To build jupyter book
  ```
  jupyter-book create darth
  ```
  More details at https://jupyterbook.org/en/stable/start/create.html

## System dependencies

  - rbenv and ruby-build

## System setup

  To get Ruby 3.0.0 working on rbenv,
  ```
  $ git -C "$(rbenv root)"/plugins/ruby-build pull
  $ rbenv install 3.0.0
  ```

  To be sure that your Ruby gems are stored only for Hesman, bundler install should be with path parameter:
  ```
  $ cd hesman
  $ bundle install --path .bundle
  ```

## Database setup (developer environment)

  Refer to `config/database.ci.yml` for database configuration.

  Your own database.yml should look something like this:
  ```
  default: &default
    adapter: postgresql
    encoding: unicode
    pool: 5

  development:
    <<: *default
    database: hesman

  test:
    <<: *default
    database: hesman_test
  ```

  To create your databases,
  ```
  $ bundle exec rails db:setup
  ```

## Services (job queues, cache servers, search engines, etc.)

## How to run the test suite
  This is a test

## Deployment instructions

Push master into the robots branch and then run the deployer:

1. Inside Hesman:
  `git checkout master && git pull && git push origin master:robots`
1. Inside Deployer:
  `rake hesman:robots:deploy`
