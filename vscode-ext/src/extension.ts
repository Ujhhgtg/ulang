import * as vscode from 'vscode';
import {
    LanguageClient,
    LanguageClientOptions,
    ServerOptions
} from 'vscode-languageclient/node';

let client: LanguageClient | undefined;

export function activate(context: vscode.ExtensionContext) {
    console.log('vscode-ulang extension is now active!');

    function startLanguageServer() {
        const config = vscode.workspace.getConfiguration('ulang');
        const executablePath = config.get<string>('executablePath') || 'ulang';

        console.log(`Starting ulang LSP server with executable: ${executablePath}`);

        const serverOptions: ServerOptions = {
            run: { command: executablePath, args: ['lsp'] },
            debug: { command: executablePath, args: ['lsp'] }
        };

        const clientOptions: LanguageClientOptions = {
            documentSelector: [{ scheme: 'file', language: 'ulang' }],
            synchronize: {
                fileEvents: vscode.workspace.createFileSystemWatcher('**/*.u')
            }
        };

        client = new LanguageClient(
            'ulangLanguageServer',
            'ulang Language Server',
            serverOptions,
            clientOptions
        );

        client.start().catch((err) => {
            vscode.window.showErrorMessage(
                `Failed to start ulang Language Server: ${err.message || err}`
            );
        });
    }

    startLanguageServer();

    context.subscriptions.push(
        vscode.workspace.onDidChangeConfiguration(async (event) => {
            if (event.affectsConfiguration('ulang.executablePath')) {
                vscode.window.showInformationMessage(
                    'ulang executable path changed. Restarting language server...'
                );
                if (client) {
                    try {
                        await client.stop();
                    } catch (e) {
                        console.error('Error stopping ulang language client:', e);
                    }
                    client = undefined;
                }
                startLanguageServer();
            }
        })
    );
}

export async function deactivate() {
    if (client) {
        await client.stop();
        client = undefined;
    }
}
