# Generated target: .github/workflows/cosmo-potctl-spike.yml
let S = import "lib/steps.ncl" in
let P = S.pins in
{
  name = "potctl Cosmopolitan APE (legacy spike alias)",
  on = {
    workflow_dispatch = {
      inputs = {
        note = {
          description = "No-op; use cosmo-potctl.yml",
          required = false,
          default = "deprecated",
        },
      },
    },
  },
  jobs = {
    redirect = {
      name = "use cosmo-potctl.yml instead",
      "runs-on" = P.ubuntu,
      steps = [
        {
          run = m%"
            echo "cosmo potctl is gating via .github/workflows/cosmo-potctl.yml"
            echo "This spike alias is a no-op (workflow_dispatch only)."
          "%,
        },
      ],
    },
  },
}
